/*
* Basic camera class
*
* Copyright (C) 2016 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "camera.h"

class CameraController
{
private:

    float znear, zfar;
    float aspect;
    float fov;

	void updateViewMatrix()
	{
        auto eye = camera->GetPosition();
        auto up = camera->GetUpVector();
        auto at = camera->GetAt();

        const RadeonProRender::float3 f = normalize(at - eye);
        const RadeonProRender::float3 s = normalize(RadeonProRender::cross(f, up));
        const RadeonProRender::float3 u = normalize(RadeonProRender::cross(s, f));

        matrices.view = glm::mat4();

        matrices.view[0][0] = s.x;
        matrices.view[1][0] = s.y;
        matrices.view[2][0] = s.z;

        matrices.view[0][1] = u.x;
        matrices.view[1][1] = u.y;
        matrices.view[2][1] = u.z;

        matrices.view[0][2] = -f.x;
        matrices.view[1][2] = -f.y;
        matrices.view[2][2] = -f.z;

        matrices.view[3][0] = -RadeonProRender::dot(s, eye);
        matrices.view[3][1] = -RadeonProRender::dot(u, eye);
        matrices.view[3][2] = RadeonProRender::dot(f, eye);

		updated = true;
	};

public:
    void LookAt(const glm::vec3 &eye, const glm::vec3 &at, const glm::vec3 &up)
    {
        camera = PerspectiveCamera::Create(
            RadeonProRender::float3(eye.x, eye.y, eye.z),
            RadeonProRender::float3(at.x, at.y, at.z),
            RadeonProRender::float3(up.x, up.y, up.z)
        );

        updateViewMatrix();
    }
    Camera::Ptr camera;


    glm::mat4 getProjection()
    {
        return matrices.perspective;
    }

    glm::mat4 getView()
    {
        return matrices.view;
    }

	enum CameraType { lookat, firstperson };
	CameraType type = CameraType::lookat;

	glm::vec3 rotation = glm::vec3();
	glm::vec3 position = glm::vec3();

	float rotationSpeed = 1.0f;
	float movementSpeed = 1.0f;

	bool updated = false;

	struct
	{
		glm::mat4 perspective;
		glm::mat4 view;
	} matrices;

	struct
	{
		bool left = false;
		bool right = false;
		bool up = false;
		bool down = false;
	} keys;

	bool moving()
	{
		return keys.left || keys.right || keys.up || keys.down;
	}

	float getNearClip() { 
		return znear;
	}

	float getFarClip() {
		return zfar;
	}

	void setPerspective(float fov, float aspect, float znear, float zfar)
	{
        this->znear = znear;
        this->zfar = zfar;
        this->aspect = aspect;
        this->fov = fov;
        const float t = tan(fov);

        matrices.perspective = glm::mat4();

        matrices.perspective[0][0] = 1.0f / (t * aspect);
        matrices.perspective[1][1] = -1.0f / t;

        matrices.perspective[2][2] = 0.0f;
        matrices.perspective[3][2] = znear;

        matrices.perspective[2][3] = -1.0f;
        matrices.perspective[3][3] = 0.0f;
	};

	void updateAspectRatio(float aspect)
	{
		//matrices.perspective = glm::perspective(glm::radians(fov), aspect, znear, zfar);
	}

	void setRotation(glm::vec3 rotation)
	{
		this->rotation = rotation;
		updateViewMatrix();
	};

	void rotate(glm::vec3 delta)
	{
		if (std::abs(delta.x) > 0.001f)
        {
            camera->Tilt(glm::radians(delta.x));
        }

        if (std::abs(delta.y) > 0.001f)
        {
            camera->Rotate(glm::radians(delta.y));
        }

		updateViewMatrix();
	}

	void setTranslation(glm::vec3 translation)
	{
		this->position = translation;
		updateViewMatrix();
	};

	void translate(glm::vec3 delta)
	{
        camera->Zoom(delta.z);

		updateViewMatrix();
	}

	void update(float deltaTime)
	{
		updated = false;
        if (moving())
        {
            float moveSpeed = deltaTime * movementSpeed;

            if (keys.up)
                camera->MoveForward(moveSpeed);
            if (keys.down)
                camera->MoveForward(-moveSpeed);
            if (keys.left)
                camera->MoveRight(moveSpeed);
            if (keys.right)
                camera->MoveRight(-moveSpeed);

            updateViewMatrix();
        }
	};

	// Update camera passing separate axis data (gamepad)
	// Returns true if view or position has been changed
	bool updatePad(glm::vec2 axisLeft, glm::vec2 axisRight, float deltaTime)
	{
		bool retVal = false;

		if (type == CameraType::firstperson)
		{
			// Use the common console thumbstick layout		
			// Left = view, right = move

			const float deadZone = 0.0015f;
			const float range = 1.0f - deadZone;

			glm::vec3 camFront;
			camFront.x = -cos(glm::radians(rotation.x)) * sin(glm::radians(rotation.y));
			camFront.y = sin(glm::radians(rotation.x));
			camFront.z = cos(glm::radians(rotation.x)) * cos(glm::radians(rotation.y));
			camFront = glm::normalize(camFront);

			float moveSpeed = deltaTime * movementSpeed * 2.0f;
			float rotSpeed = deltaTime * rotationSpeed * 50.0f;
			 
			// Move
			if (fabsf(axisLeft.y) > deadZone)
			{
				float pos = (fabsf(axisLeft.y) - deadZone) / range;
				position -= camFront * pos * ((axisLeft.y < 0.0f) ? -1.0f : 1.0f) * moveSpeed;
				retVal = true;
			}
			if (fabsf(axisLeft.x) > deadZone)
			{
				float pos = (fabsf(axisLeft.x) - deadZone) / range;
				position += glm::normalize(glm::cross(camFront, glm::vec3(0.0f, 1.0f, 0.0f))) * pos * ((axisLeft.x < 0.0f) ? -1.0f : 1.0f) * moveSpeed;
				retVal = true;
			}

			// Rotate
			if (fabsf(axisRight.x) > deadZone)
			{
				float pos = (fabsf(axisRight.x) - deadZone) / range;
				rotation.y += pos * ((axisRight.x < 0.0f) ? -1.0f : 1.0f) * rotSpeed;
				retVal = true;
			}
			if (fabsf(axisRight.y) > deadZone)
			{
				float pos = (fabsf(axisRight.y) - deadZone) / range;
				rotation.x -= pos * ((axisRight.y < 0.0f) ? -1.0f : 1.0f) * rotSpeed;
				retVal = true;
			}
		}
		else
		{
			// todo: move code from example base class for look-at
		}

		if (retVal)
		{
			updateViewMatrix();
		}

		return retVal;
	}

};

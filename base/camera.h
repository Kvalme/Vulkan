/**********************************************************************
 Copyright (c) 2016 Advanced Micro Devices, Inc. All rights reserved.
 
 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:
 
 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.
 
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 ********************************************************************/

/**
 \file camera.h
 \author Dmitry Kozlov
 \version 1.0
 \brief Contains declaration of camera types supported by the renderer.
 */
#pragma once

#include <memory>

#include "Rpr/Math/float3.h"
#include "Rpr/Math/float2.h"
#include "Rpr/Math/quaternion.h"

class Camera
{
public:
    using Ptr = std::shared_ptr<Camera>;

    Camera() = default;
    virtual ~Camera() = 0;

    struct
    {
        bool left = false;
        bool right = false;
        bool up = false;
        bool down = false;
    } keys;

    // Pass camera position, camera aim, camera up vector
    void LookAt(RadeonProRender::float3 const& eye,
                RadeonProRender::float3 const& at,
                RadeonProRender::float3 const& up);

    // Rotate camera around world Z axis
    void Rotate(float angle);

    // Tilt camera
    void Tilt(float angle);
    // Move along camera Z direction
    void MoveForward(float distance);
    void Zoom(float distance);
    // Move along camera X direction
    void MoveRight(float distance);
    // Move along camera Y direction
    void MoveUp(float distance);
    // Move to position eye and look at where q is pointing;  used for HMD input
    void RotateAndMoveTo(RadeonProRender::quaternion q, RadeonProRender::float3 eye);

    RadeonProRender::float3 GetForwardVector() const;
    RadeonProRender::float3 GetUpVector() const;
    RadeonProRender::float3 GetRightVector() const;
    RadeonProRender::float3 GetPosition() const;
    RadeonProRender::float3 GetAt() const;

    // Set camera depth range.
    // Does not really make sence for physical camera
    void SetDepthRange(RadeonProRender::float2 const& range);
    RadeonProRender::float2 GetDepthRange() const;

    float GetAspectRatio() const;

    // Set camera sensor size in meters.
    // This distinguishes APC-S vs full-frame, etc
    void SetSensorSize(RadeonProRender::float2 const& size);
    RadeonProRender::float2 GetSensorSize() const;

protected:
    // Rotate camera around world Z axis
    void Rotate(RadeonProRender::float3, float angle);
    void RotateOnOrbit(RadeonProRender::float3, float angle);

    Camera(RadeonProRender::float3 const& eye,
            RadeonProRender::float3 const& at,
            RadeonProRender::float3 const& up);

    // Camera coordinate frame
    RadeonProRender::float3 m_forward;
    RadeonProRender::float3 m_right;
    RadeonProRender::float3 m_up;
    RadeonProRender::float3 m_p;
    RadeonProRender::float3 m_at;

    // Image plane width & hight in scene units
    RadeonProRender::float2 m_dim;

    // Near and far Z
    RadeonProRender::float2 m_zcap;

};

class PerspectiveCamera : public Camera
{
public:
    using Ptr = std::shared_ptr<PerspectiveCamera>;
    static Ptr Create(RadeonProRender::float3 const& eye,
                        RadeonProRender::float3 const& at,
                        RadeonProRender::float3 const& up);


    // Set camera focus distance in meters,
    // this is essentially a distance from the lens to the focal plane.
    // Altering this is similar to rotating the focus ring on real lens.
    void SetFocusDistance(float distance);
    float GetFocusDistance() const;

    // Set camera focal length in meters,
    // this is essentially a distance between a camera sensor and a lens.
    // Altering this is similar to rotating zoom ring on a lens.
    void SetFocalLength(float length);
    float GetFocalLength() const;

    // Set aperture value in meters.
    // This is a radius of a lens.
    void SetAperture(float aperture);
    float GetAperture() const;

protected:
    // Pass camera position, camera aim, camera up vector, depth limits, vertical field of view
    // and image plane aspect ratio
    PerspectiveCamera(RadeonProRender::float3 const& eye,
                        RadeonProRender::float3 const& at,
                        RadeonProRender::float3 const& up);

private:
    float  m_focal_length;
    float  m_focus_distance;
    float  m_aperture;

    friend std::ostream& operator << (std::ostream& o, PerspectiveCamera const& p);
};

inline Camera::~Camera()
{
}

inline void PerspectiveCamera::SetFocusDistance(float distance)
{
    m_focus_distance = distance;
}

inline void PerspectiveCamera::SetFocalLength(float length)
{
    m_focal_length = length;
}

inline void PerspectiveCamera::SetAperture(float aperture)
{
    m_aperture = aperture;
}

inline float PerspectiveCamera::GetFocusDistance() const
{
    return m_focus_distance;
}

inline float PerspectiveCamera::GetFocalLength() const
{
    return m_focal_length;
}

inline float PerspectiveCamera::GetAperture() const
{
    return m_aperture;
}

inline RadeonProRender::float2 Camera::GetSensorSize() const
{
    return m_dim;
}

inline void Camera::SetSensorSize(RadeonProRender::float2 const& size)
{
    m_dim = size;
}

inline void Camera::SetDepthRange(RadeonProRender::float2 const& range)
{
    m_zcap = range;
}

inline RadeonProRender::float2 Camera::GetDepthRange() const
{
    return m_zcap;
}

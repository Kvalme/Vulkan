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
#include "camera.h"

#include <cmath>
#include <cassert>

#include "Rpr/Math/float3.h"
#include "Rpr/Math/float2.h"
#include "Rpr/Math/quaternion.h"
#include "Rpr/Math/matrix.h"
#include "Rpr/Math/mathutils.h"

using namespace RadeonProRender;

Camera::Camera(float3 const& eye, float3 const& at, float3 const& up)
{
    LookAt(eye, at, up);
}

PerspectiveCamera::PerspectiveCamera(float3 const& eye, float3 const& at, float3 const& up)
: Camera(eye, at, up)
, m_focal_length(0.f)
, m_focus_distance(0.f)
, m_aperture(0.f)
{
}

void Camera::LookAt(float3 const& eye,
                    float3 const& at,
                    float3 const& up)
{
    m_p = eye;
    m_forward = normalize(at - eye);
    m_right = normalize(cross(m_forward, up));
    m_up = cross(m_right, m_forward);
    m_at = at;
}

// Rotate camera around global Y axis, use for FPS camera
void Camera::Rotate(float angle)
{
    auto axis = float3(0, 1, 0);
    if (dot(axis, m_up) < 0)
    {
        axis = -axis;
    }
    auto q = rotation_quaternion(axis, angle);
    auto new_direction = rotate_vector(m_at - m_p, q);
    m_at = m_p + new_direction;
    m_forward = normalize(new_direction);
    m_right = normalize(rotate_vector(m_right, q));
    m_up = cross(m_right, m_forward);
}

// Rotate camera around local X axis, use for FPS camera
void Camera::Tilt(float angle)
{
    auto q = rotation_quaternion(m_right, angle);
    auto new_direction = rotate_vector(m_at - m_p, q);
    m_at = m_p + new_direction;
    m_forward = normalize(new_direction);
    m_up = cross(m_right, m_forward);
}

void Camera::Rotate(float3 v, float angle)
{
    auto q = rotation_quaternion(v, angle);
    auto new_direction = rotate_vector(m_at - m_p, q);
    auto new_up = rotate_vector(m_up, q);

    LookAt(m_p, m_p + new_direction, new_up);
}

void Camera::RotateOnOrbit(float3 v, float angle)
{
    /// matrix should have basis vectors in rows
    /// to be used for quaternion construction
    /// would be good to add several options
    /// to quaternion class

    // Rotate camera frame around v
    auto q = rotation_quaternion(v, angle);
    auto new_direction = rotate_vector(m_p - m_at, q);
    auto new_up = rotate_vector(m_up, q);

    LookAt(m_at + new_direction, m_at, new_up);
}

void Camera::RotateAndMoveTo(quaternion q, float3 eye)
{
    float3 up = rotate_vector(float3(0, 1.0f, 0), q);
    m_p = eye;
    m_forward = rotate_vector(float3(0, 0, -1.0f), q);
    m_at = m_forward + eye;
    m_right = normalize(cross(m_forward, up));
    m_up = up;
}

// Move along camera Z direction
void Camera::MoveForward(float distance)
{
    m_p += distance * m_forward;
    m_at += distance * m_forward;
}

void Camera::Zoom(float distance)
{
    if (distance < 0.f || (m_p - m_at).sqnorm() > 1.f)
    {
        m_p += distance * m_forward;
        LookAt(m_p, m_at, m_up);
    }
}

// Move along camera X direction
void Camera::MoveRight(float distance)
{
    m_p += distance * m_right;
    m_at += distance * m_right;
}

// Move along camera Y direction
void Camera::MoveUp(float distance)
{
    m_p += distance * m_up;
    m_at += distance * m_up;
}

float3 Camera::GetForwardVector() const
{
    return m_forward;
}

float3 Camera::GetUpVector() const
{
    return m_up;
}

float3 Camera::GetRightVector() const
{
    return m_right;
}

float3 Camera::GetPosition() const
{
    return m_p;
}
float3 Camera::GetAt() const
{
    return m_at;
}
float Camera::GetAspectRatio() const
{
    return m_dim.x / m_dim.y;
}

namespace {
    struct PerspectiveCameraConcrete : public PerspectiveCamera {
        PerspectiveCameraConcrete(float3 const& eye, float3 const& at, float3 const& up) :
        PerspectiveCamera(eye, at, up) {}
    };
}

PerspectiveCamera::Ptr PerspectiveCamera::Create(float3 const& eye,
            float3 const& at,
            float3 const& up) {
    return std::make_shared<PerspectiveCameraConcrete>(eye, at, up);

}

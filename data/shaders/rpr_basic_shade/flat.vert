#version 450

layout (location = 0) in vec3 inPosition;

layout (push_constant) uniform PushConstants
{
    mat4 g_transform;
	vec4 g_ambient_color;
};

out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
    gl_Position = g_transform * vec4(inPosition, 1.0);
}
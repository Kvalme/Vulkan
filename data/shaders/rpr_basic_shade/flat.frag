#version 450

layout (location = 0) out vec4 color;

layout (push_constant) uniform PushConstants
{
    mat4 g_transform;
	vec4 g_ambient_color;
};

void main()
{
    color = g_ambient_color;
}

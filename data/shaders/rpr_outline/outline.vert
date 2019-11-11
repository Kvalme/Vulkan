#version 450

layout (location = 0) in vec3 inPosition;

layout (location = 0) out vec2 screen_coord;

layout (push_constant) uniform PushConstants
{
    mat4 g_transform;
	vec4 g_outline_color;
	vec2 g_tex_coord_offset;
};

out gl_PerVertex
{
    vec4 gl_Position;
};

void main()
{
	vec4 p = g_transform * vec4(inPosition, 1.0);
	gl_Position = p;
	screen_coord = (p.xy / p.w) * 0.5 + 0.5;
}
#version 450

layout (location = 0) in vec2 tex_coord;

out float gl_FragDepth;

layout (binding = 0) uniform sampler2D tex;

layout (push_constant) uniform PushConstants
{
    vec2 g_projection_coefficients;
    float g_depth_range;
};

void main()
{
    float linear_depth = texture(tex, tex_coord).x;
    float remapped_linear_depth = linear_depth * g_depth_range;
    float hardware_depth = g_projection_coefficients.y / (remapped_linear_depth - g_projection_coefficients.x);
    gl_FragDepth = hardware_depth;
}

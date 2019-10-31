#version 450

#extension GL_ARB_separate_shader_objects : enable
#extension GL_ARB_shading_language_420pack : enable
#extension GL_GOOGLE_include_directive : enable

const vec3 vertices[] = 
{ 
    {1.0, 0.0, 0.0},
    {0.0, 1.0, 0.0},
    {0.0, 0.0, 1.0},
};

layout (binding = 0) uniform UBO
{
	mat4 model;
    mat4 view;
	mat4 projection;
    uint is_visible;
    float scale;
} ubo;

layout (location = 0) out vec3 out_color;

out gl_PerVertex
{
    vec4 gl_Position;
};

// Run this shader for 6 vertices without vertex and index buffers
// and in line list mode.

void main()
{
    out_color = vertices[gl_VertexIndex >> 1];
    
    vec3 vert = vec3(0.0);
    
    if (ubo.is_visible > 0 && (gl_VertexIndex & 1) == 1)
    {
        vert = vertices[gl_VertexIndex >> 1] * vec3(ubo.scale);
    }
    
    gl_Position = ubo.projection * ubo.view * ubo.model * vec4(vert.xyz, 1.0);
}

#version 450

#define OUTLINE_RADIUS 4

layout(location = 0) in vec2 screen_coord;

layout (location = 0) out vec4 color;

layout (binding = 0) uniform sampler2D shape_id_aov;

layout (push_constant) uniform PushConstants
{
    mat4 g_transform;
	vec4 g_outline_color;
	vec2 g_tex_coord_offset;
};

void main()
{
	float ref_shape_id = textureLod(shape_id_aov, screen_coord, 0).x;
	
	bool pixel_is_outline = false;
	
	for (int j = -OUTLINE_RADIUS; j <= OUTLINE_RADIUS; ++j)
	{
		for (int i = -OUTLINE_RADIUS; i <= OUTLINE_RADIUS; ++i)
		{
			if (((i == 0) && (j == 0)) ||
				(i * i + j * j > OUTLINE_RADIUS * OUTLINE_RADIUS))
			{
				continue;
			}
			
			float pixel_shape_id = textureLod(shape_id_aov, screen_coord + vec2(i, j) * g_tex_coord_offset, 0).x;
			pixel_is_outline = pixel_is_outline || (pixel_shape_id != ref_shape_id);
		}
	}
	
	if (!pixel_is_outline)
	{
		discard;
	}
	
	color = g_outline_color;
}

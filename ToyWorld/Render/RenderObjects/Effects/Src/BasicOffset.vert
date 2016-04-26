#version 330


// Texture dimensions in px, tiles per row
uniform ivec3	texSizeCount	= ivec3(256,256, 16);
// Tile size, tile margin in px
uniform ivec4	tileSizeMargin	= ivec4(16,16, 0,0);

uniform mat4 mvp = mat4(1);


layout(location=0) in vec2	v_position;
layout(location=1) in int	v_texOffset;

out vec2 f_texCoods;


vec2 GetTexCoods()
{
	// Tile positions
    ivec2 off = ivec2(v_texOffset % texSizeCount.z, v_texOffset / texSizeCount.z);
	// Texture positions (top-left)
	vec2 uv = off * (tileSizeMargin.xy + tileSizeMargin.zw);

	// Offset the vertex according to its position in the quad
	int vertID = gl_VertexID % 4;

	switch (vertID)
	{
		case 0:
		uv += ivec2(0, tileSizeMargin.y);
		break;
				
		case 2:
		uv += ivec2(tileSizeMargin.x, 0);
		break;
				
		case 3:
		uv += tileSizeMargin.xy;
		break;
	}

	// Normalize to tex coordinates
	return vec2(uv) / texSizeCount.xy;
}

void main()
{
	if (v_texOffset <= 0)
	{
		// If this vertex is a part of a quad that does not contain any tile to display, set it to a default position to discard it
		gl_Position = vec4(0,0,2000,0);
		f_texCoods = vec2(0,0);
		return;
	}
		
	f_texCoods = GetTexCoods();
	gl_Position = mvp * vec4(v_position, 0, 1);
}

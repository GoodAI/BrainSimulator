#version 330

// Texture dimensions in px, tiles per row/col
uniform ivec4	texSizeCount	= ivec4(256,256, 16,16);
// Tile size, tile margin in px
uniform ivec4	tileSizeMargin	= ivec4(16,16, 1,1);

layout(location=0) in vec2	v_position;
layout(location=1) in int	v_texOffset;

out vec2 f_texCoods;


vec2 GetTexCoods()
{
	// Tile positions
    ivec2 off = ivec2(v_texOffset % texSizeCount.z, v_texOffset / texSizeCount.w);
	// Texture positions (bot-left)
	ivec2 pos = off * (tileSizeMargin.xy + tileSizeMargin.zw);

	// Offset the vertex according to its position in the quad
	switch (gl_VertexID % 4)
	{
		case 1:
		off = ivec2(tileSizeMargin.x, 0);
		break;
				
		case 2:
		off = ivec2(0, tileSizeMargin.y);
		break;
				
		case 3:
		off = ivec2(tileSizeMargin.x, tileSizeMargin.y);
		break;
	}

	pos += off;

	// Normalize to tex coordinates
	return pos / texSizeCount.xy;
}

void main()
{
	//f_texCoods = GetTexCoods();
	f_texCoods = vec2(v_texOffset / 9.0f, 0);
	gl_Position = vec4(v_position, 0, 2);
}

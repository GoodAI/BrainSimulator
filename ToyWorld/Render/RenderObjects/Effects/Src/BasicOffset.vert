#version 330


const int MODULO_BITS = 12;
const int MODULO_MASK = (1 << MODULO_BITS) - 1; // 0x00000FFF


// Texture dimensions in px, tiles per row
uniform ivec3	texSizeCount = ivec3(256, 256, 16);
// Tile size, tile margin in px
uniform ivec4	tileSizeMargin = ivec4(16, 16, 0, 0);
// Tile border size increase after tileset preprocessing
uniform ivec2   tileBorder = ivec2(2, 2);

uniform mat4 mvp = mat4(1);


layout(location = 0) in vec2	v_position;
layout(location = 1) in int		v_texOffset;

smooth out vec2 f_texCoods;
flat out int f_samplerIdx;


vec2 GetTexCoods(int tileOffset)
{
	// Tile positions
	ivec2 off = ivec2(tileOffset % texSizeCount.z, tileOffset / texSizeCount.z);
	// Texture positions (top-left)
	vec2 tileSize = tileSizeMargin.xy + tileSizeMargin.zw + tileBorder * 2;
	vec2 uv = off * tileSize + tileBorder;
	// + tileBorder because even the first tile's border size was increased

	// Offset the vertex according to its position in the quad
	int vertID = gl_VertexID % 4;

	const float offset = 0.02f;
	vec2 uvOffset = vec2(offset, -offset);

	switch (vertID)
	{
	case 0:
		uv += ivec2(0, tileSizeMargin.y);
		uv += uvOffset.xy;
		break;

	case 1:
		uv += uvOffset.xx;
		break;

	case 2:
		uv += ivec2(tileSizeMargin.x, 0);
		uv += uvOffset.yx;
		break;

	case 3:
		uv += tileSizeMargin.xy;
		uv += uvOffset.yy;
		break;
	}

	// Normalize to tex coordinates
	return vec2(uv) / texSizeCount.xy;
}

void main()
{
	int tileOffset = v_texOffset & MODULO_MASK; // It's the same as v_texOffset % (MODULO_MASK + 1)

	if (tileOffset <= 0)
	{
		// If this vertex is a part of a quad that does not contain any tile to display, set it to a default position to discard it
		gl_Position = vec4(0, 0, 2000, 0);
		f_texCoods = vec2(0, 0);
		return;
	}

	f_texCoods = GetTexCoods(tileOffset);
	f_samplerIdx = v_texOffset / (1 << MODULO_BITS); // It's the same as v_texOffset / (MODULO_MASK + 1)
	f_samplerIdx = 1;
	gl_Position = mvp * vec4(v_position, 0, 1);
}

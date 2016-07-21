#version 330


const int MODULO_BITS = 12;
const int MODULO_MASK = (1 << MODULO_BITS) - 1; // 0x00000FFF


uniform usampler1D tileTypesTexture;

// Texture dimensions in px, tiles per row
uniform ivec3	texSizeCount =		ivec3(256, 256, 16);
// Tile size, tile margin in px
uniform ivec4	tileSizeMargin =	ivec4(16, 16, 0, 0);
// Tile border size increase after tileset preprocessing
uniform ivec2   tileBorder =		ivec2(2, 2);

uniform mat4 mvp = mat4(1);


layout(location = 0) in vec3	v_position;
layout(location = 1) in unsigned short		v_texOffset;

smooth out	vec2	f_texCoods;
flat out	int		f_samplerIdx;


vec2 GetTexCoods(int tileOffset)
{
	// Tiles are indexed from 1.......
	tileOffset--;

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
	int tileType = int(v_texOffset);// int(texelFetch(tileTypesTexture, gl_VertexID / 8, 0).r);
	int tileOffset = tileType & MODULO_MASK; // It's the same as v_texOffset % (MODULO_MASK + 1)

	//if (tileOffset <= 1) // Tiles are indexed from 1......
	//{
	//	// If this vertex is a part of a quad that does not contain any tile to display, set it to a default position to discard it
	//	gl_Position = vec4(0, 0, 2000, 0);
	//	f_samplerIdx = 64;
	//	return;
	//}

	f_texCoods = GetTexCoods(tileOffset);
	f_samplerIdx = tileType >> MODULO_BITS; // It's the same as v_texOffset / (MODULO_MASK + 1)
	gl_Position = mvp * vec4(v_position, 1);
}

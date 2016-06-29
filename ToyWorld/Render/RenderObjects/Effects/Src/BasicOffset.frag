#version 330


uniform sampler2D tilesetTexture;
uniform sampler2D tilesetTextureWinter;

// Lighting coefficients
// Ambient color and coefficient
uniform vec4 ambient = vec4(vec3(1), 0.25f);
// Diffuse color and coefficient
uniform vec4 diffuse = vec4(vec3(1), 0.75f);


smooth in vec2 f_texCoods;
flat in int f_samplerIdx;

layout(location = 0) out vec4 out_color;


void main()
{
	switch (f_samplerIdx)
	{
	case 0:
		out_color = texture(tilesetTexture, f_texCoods);
		break;
	case 1:
		out_color = texture(tilesetTextureWinter, f_texCoods);
		break;
	default:
		out_color = vec4(1, 0, 0, 1);
		break;
	}
	
	out_color.xyz *= ambient.w * ambient.xyz + diffuse.w * diffuse.xyz;
}

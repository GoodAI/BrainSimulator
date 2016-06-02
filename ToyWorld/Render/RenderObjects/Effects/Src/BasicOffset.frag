#version 330


uniform sampler2D tilesetTexture;

// Lighting coefficients
// Ambient color and coefficient
uniform vec4 ambient = vec4(vec3(1), 0.25f);
// Diffuse color and coefficient
uniform vec4 diffuse = vec4(vec3(1), 0.75f);


smooth in vec2 f_texCoods;

layout(location = 0) out vec4 out_color;


void main()
{
	out_color = texture(tilesetTexture, f_texCoods);
	out_color.xyz = (ambient.w * ambient.xyz + diffuse.w * diffuse.xyz) * out_color.xyz;
}

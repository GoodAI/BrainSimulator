#version 330


uniform sampler2D tilesetTexture;

// Lighting coefficients
// Ambient color and coefficient
uniform vec4 ambient = vec4(vec3(255), 0.25f);
// Diffuse color and coefficient
uniform vec4 diffuse = vec4(vec3(255), 0.75f);


smooth in vec2 f_texCoods;

layout(location = 0) out vec4 out_color;


void main()
{
	out_color = texture(tilesetTexture, f_texCoods);
	vec3 ambientNorm = ambient.xyz / 255;
	vec3 diffuseNorm = diffuse.xyz / 255;
	out_color.xyz = (ambient.w * ambientNorm + diffuse.w * diffuseNorm) * out_color.xyz;
}

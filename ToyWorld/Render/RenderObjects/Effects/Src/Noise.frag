#version 330


uniform vec4 noiseColor = vec4(1,1,1,1);
uniform float time = 1;


smooth in vec2 f_worldPos;

layout(location=0) out vec4 out_color;


float cnoise(vec3 P);

void main()
{
	float noise = cnoise(vec3(f_worldPos, time)); // Should be between [-1,1]
	out_color = noiseColor;
	out_color.z *= (noise / 2 + 0.5f);
}

#version 330


uniform vec4 noiseColor = vec4(1, 1, 1, 1);
uniform float time = 1;


smooth in vec2 f_worldPos;

layout(location = 0) out vec4 out_color;


float cnoise(vec3 P);

void main()
{
	float noise = 1 + cnoise(vec3(f_worldPos, time)); // Should be between (-1,1), offset to (0,2)
	const float mean = 0.8f;

	if (noise < 1f)
		// scale (0,1) to (0,mean)
		noise *= mean;
	else
		// offset (1,2) to (0,1), scale to (1-mean), offset to (mean,1)
		noise = (noise - 1f) * (1 - mean) + mean;

	out_color = noiseColor * noise;
}

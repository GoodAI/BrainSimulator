#version 330


uniform sampler2D sceneTexture;

uniform vec4 noiseColor = vec4(1, 1, 1, 1);
// Time			  -- third dimension for perlin noise
// Step			  -- amount between time steps
// Mean			  -- noise results are in interval (0,2), values are scaled from (0,1) to (0,mean) and from (1,2) to (mean,1)
uniform vec4 timeMean = vec4(0, 0.01f, 0.8f, 0);


smooth in vec2 f_worldPos;

layout(location = 0) out vec4 out_color;


float cnoise(vec3 P);

void main()
{
	float noise = 1 + cnoise(vec3(f_worldPos, timeMean.x)); // Should be between (-1,1), offset to (0,2)
	float mean = timeMean.z;

	if (noise < 1f)
		// scale (0,1) to (0,mean)
		noise *= mean;
	else
		// offset (1,2) to (0,1), scale to (1-mean), offset to (mean,1)
		noise = (noise - 1f) * (1 - mean) + mean;

	out_color = noiseColor;
	out_color.w *= noise;
}

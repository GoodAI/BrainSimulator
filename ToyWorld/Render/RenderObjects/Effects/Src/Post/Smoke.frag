#version 330


uniform vec4 smokeColor = vec4(1, 1, 1, 1);
// Time			  -- third dimension for perlin noise
// Step			  -- amount between time steps
// MeanCoef		  -- noise results are in interval (0,2), values are scaled from (0,1) to (0,mean) and from (1,2) to (mean,1)
// ScaleCoef	  -- the "zoom" of the noise
uniform vec4 timeMeanScale = vec4(0, 0.01f, 1, 1);


smooth in vec2 f_worldPos;

layout(location = 0) out vec4 out_color;


float snoise(vec3 P);

void main()
{
	float noise = 1 + snoise(vec3(f_worldPos * 0.6f * timeMeanScale.w, timeMeanScale.x)); // Should be between (-1,1), offset to (0,2)
	float mean = 0.6f * timeMeanScale.z;

	if (noise < 1f)
		// scale (0,1) to (0,mean)
		noise *= mean;
	else
		// offset (1,2) to (0,1), scale to (1-mean), offset to (mean,1)
		noise = (noise - 1f) * (1 - mean) + mean;

	out_color.w = smokeColor.w * noise;
	out_color.xyz = smokeColor.xyz * out_color.w; // we are using pre-multiplied alpha blending
}

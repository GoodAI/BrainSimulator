#version 330


uniform sampler2D sceneTexture;

uniform ivec2 viewportSize = ivec2(1024, 1024);

// Time			  -- third dimension for perlin noise
// Step			  -- amount between time steps
uniform vec2 timeStep = vec2(0, 0.01f);
// Variance		  -- the noise variance
uniform float variance = 1;


smooth in vec2 f_worldPos;

layout(location = 0) out vec4 out_color;


float rand(vec2 co);
vec2 gaussrand(vec2 randoms);

void main()
{
	float fragIdx = gl_FragCoord.y * viewportSize.x + gl_FragCoord.x;

	vec4 rands;
	float step = timeStep.y * 0.2f; // pre-multiply step -- we want to get 4 unique numbers within the interval (timestep.x, timestep.x + timestep.y)
	vec2 seed = vec2(timeStep.x, rand(vec2(fragIdx, fragIdx))); // scatter fragIdx to reduce artifacts

	// Generate 4 uniform randoms, use unique seed from within the timestep interval
	for (int i = 0; i < 4; ++i)
	{
		rands[i] = rand(seed);
		seed.x += step;
	}

	// Generate 4 gaussian randoms
	vec4 gaussRands = vec4(gaussrand(rands.xy), gaussrand(rands.zw)) * variance * 0.1f;

	out_color = texture(sceneTexture, gl_FragCoord.xy / viewportSize);
	out_color.xyz = out_color.xyz + gaussRands.xyz;
}

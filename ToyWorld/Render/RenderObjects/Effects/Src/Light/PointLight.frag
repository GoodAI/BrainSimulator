#version 330


// Lighting coefficients
// Light color and intensity
uniform vec4 colorIntensity = vec4(vec3(220), 0.5f);
// The decay rate of the light
uniform float decay = 50;
// The light's world position
uniform vec3 lightPos = vec3(0);


smooth in vec3 f_worldPos;

layout(location = 0) out vec4 out_color;


void main()
{
	vec3 dist = lightPos - f_worldPos;

	float intensity = 1 / (1 + decay * dot(dist, dist));

	out_color = vec4(colorIntensity.xyz, 220);
	out_color *= intensity * colorIntensity.w;
	out_color.xyz *= out_color.w; // Pre-multiply by alpha
}

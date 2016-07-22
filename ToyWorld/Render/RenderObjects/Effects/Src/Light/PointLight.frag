#version 330


// Lighting coefficients
// Light's color
uniform vec4 color = vec4(0.85f);
// The intensity and decay rate of the light
uniform vec2 intensityDecay = vec2(0.3f, 1);
// The light's world position
uniform vec3 lightPos = vec3(0);


smooth in vec3 f_worldPos;

layout(location = 0) out vec4 out_color;


void main()
{
	vec3 dist = lightPos - f_worldPos;

	float intensity = 1 / (1 + dot(dist, dist) * 30 / intensityDecay.y);
	intensity = 255 * 0.3f * intensity * intensityDecay.x;

	if (intensity < 0.05f)
		discard;

	out_color = color * intensity;
	out_color.xyz *= out_color.w; // Pre-multiply by alpha
}

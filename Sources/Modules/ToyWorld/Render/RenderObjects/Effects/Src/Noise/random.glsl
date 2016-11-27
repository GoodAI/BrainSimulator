// Modified one-liner rand version
// http://byteblacksmith.com/improvements-to-the-canonical-one-liner-glsl-rand-for-opengl-es-2-0/
float rand(vec2 co)
{
	float a = 12.9898;
	float b = 78.233;
	float c = 43758.5453;
	float dt = dot(co.xy, vec2(a, b));
	float sn = mod(dt, 3.14);
	return fract(sin(sn) * c);
}


#define M_PI 3.1415926535897932384626433832795

// * 0.9998 + 0.0001: to avoid numerical instability when dealing with random numbers too close to 0 or 1
#define STABILIZE(x) ((x) * 0.9998 + 0.0001)

// Box-Muller method for sampling from the normal distribution
// http://en.wikipedia.org/wiki/Normal_distribution#Generating_values_from_normal_distribution
// This method requires 2 uniform [0,1] random inputs and produces 2 
vec2 gaussrand(vec2 randoms)
{
	float v1 = sqrt(-2.0 * log(STABILIZE(randoms.x)));
	float v2 = 2.0 * M_PI * STABILIZE(randoms.y);

	return vec2(
		v1 * sin(v2),
		v1 * cos(v2));
}


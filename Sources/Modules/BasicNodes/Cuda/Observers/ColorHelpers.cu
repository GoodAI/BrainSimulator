#define _SIZE_T_DEFINED 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __cplusplus 
#define __cplusplus 
#endif

#include <cuda.h> 
#include <texture_fetch_functions.h> 
#include <float.h>

#define _2_INV_PI 0.63662f


#define GET_WHITE ((100 << 24)  | (255 << 16) | (255 << 8) | 255)
#define GET_RGBA(r,g,b,a) ( ((a) << 24)  | ((r) << 16) | ((g) << 8) | (b) )
#define GET_GREY(value) ( ((100) << 24)  | ((value) << 16) | ((value) << 8) | (value) )

#define RGB2GRAY_AVERAGE(r,g,b) ( ((r)+(g)+(b)) / 3 )



__device__ void getRGBfromChar(int rgb, int & r, int & g, int & b){
		r = (rgb >> 16) & 255;
		g = (rgb >> 8) & 255;
		b = rgb & 255;
	}
__device__ int weightColor(int color, float weight){
		return (int) ( (float)color * weight);
	}

__device__ float scale_to_interval(float x, float min, float max) {
	if (min >= max)
		return 0;

	x = fmaxf(min, fminf(x, max));

	x -= min;

	x /= (max - min);

	return x;
}


__device__ void hsv_to_rgb(float h, float s, float v, float &r, float &g, float &b) {
	float f = h * 6;
	float hi = floorf(f);
	f = f - hi;
	float p = v * (1 - s);
	float q = v * (1 - s * f);
	float t = v * (1 - s * (1 - f));

	if(hi == 0.0f || hi == 6.0f) {
		r = v;
		g = t;
		b = p;
	} else if(hi == 1.0f) {
		r = q;
		g = v;
		b = p;
	} else if(hi == 2.0f) {
		r = p;
		g = v; 
		b = t;
	} else if(hi == 3.0f) {
		r = p;
		g = q;
		b = v;
	} else if(hi == 4.0f) {
		r = t;
		g = p;
		b = v;
	} else {
		r = v;
		g = p;
		b = q;
	}
}

__device__ unsigned int hsva_to_uint_rgba(float h, float s, float v, float a) {
	float r, g, b;	

    hsv_to_rgb(h, s, v, r, g, b);

	unsigned char red = (unsigned char) __float2uint_rn(255.0f * r);
	unsigned char green = (unsigned char) __float2uint_rn(255.0f * g);
	unsigned char blue = (unsigned char) __float2uint_rn(255.0f * b);
	unsigned char alpha = (unsigned char) __float2uint_rn(255.0f * a);	

	return  (alpha << 24)  | (red << 16) | (green << 8) | blue;			
}

__device__ float hsva_to_float(float h, float s, float v) {
	float r, g, b;	

    hsv_to_rgb(h, s, v, r, g, b);

	return  RGB2GRAY_AVERAGE(r,g,b);			
}

__device__ unsigned int rgba_to_uint_rgba(float r, float g, float b, float a) {

	unsigned char red = (unsigned char) __float2uint_rn(255.0f * r);
	unsigned char green = (unsigned char) __float2uint_rn(255.0f * g);
	unsigned char blue = (unsigned char) __float2uint_rn(255.0f * b);
	unsigned char alpha = (unsigned char) __float2uint_rn(255.0f * a);	

	return  (alpha << 24)  | (red << 16) | (green << 8) | blue;			
}

__device__ unsigned int grayscale_to_uint_rgba(float value)
{
	return rgba_to_uint_rgba(value, value, value, 1.0f);
}

__device__ unsigned int float_to_uint_rgba(float input, int method, int scale, float minValue, float maxValue)
{				
	if (method == 5) // RAW data
	{
		return *((unsigned int *)&input);
	}
	else
	{
		float x = input;

		if (scale == 0) 
		{
			if (method == 0 && minValue < 0 && maxValue > 0) {

				float absx = fabsf(x);					
				x = (x + absx) / maxValue * 0.5 + (absx - x) / minValue * 0.5; 					
			}
			else 
			{
				x = (x - minValue) / (maxValue - minValue);
			}
		}
		else if (scale == 1) 
		{
			x = atanf(x / maxValue) * _2_INV_PI;
		}	
		else if (scale == 2) 
		{
			x = x / (maxValue + sqrtf(maxValue + x * x));
		}

		float hue = 0;
		float saturation = 0;
		float value = 1.0f;

		if (isnan(input)) 
		{
			return rgba_to_uint_rgba(0, 0, 1.0f, 1.0f);
		}
		else if (isinf(input) && input > 0) 
		{
			return rgba_to_uint_rgba(0, 1.0f, 1.0f, 1.0f);
		}
		else if (isinf(input) && input < 0) 
		{
			return rgba_to_uint_rgba(1.0f, 0, 1.0f, 1.0f);
		}
		else {
			if (method == 0) {
				hue = (1 - signbit(x)) * 0.33203125f;
				saturation = 1;
				value = fminf(fabsf(x), 1);
			}
			if (method == 1) {	
				value = fminf(fmaxf(x, 0), 1);
			}
			else if (method == 2) {
				hue = (1.0f - fminf(fmaxf(x, 0), 1)) * 0.671875f;
				saturation = 1.0f;
			}	
			else if (method == 3) {
				value = fminf(fmaxf(x, 0), 1) > 0.5;				
			}

			return hsva_to_uint_rgba(hue, saturation, value, 1.0);
		}
	}	
}
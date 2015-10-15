//
//
//
// includes for observers.
//
//
//

#define _SIZE_T_DEFINED 

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>
#include "math.h"

#include "../Observers/ColorHelpers.cu"


//#define GET_WHITE ((100 << 24)  | (255 << 16) | (255 << 8) | 255)
//#define GET_RGBA(r,g,b,a) ( ((a) << 24)  | ((r) << 16) | ((g) << 8) | (b) )


extern "C"  
{	





	




	__global__ void FillVBOFromInputImage (float* values, int imageSize, unsigned int* pixels){
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		if (id<imageSize){
            float fval = fminf(fmaxf(values[id], 0), 1);
			int val = fval*255;
			pixels[id] = GET_RGBA(val,val,val,255);
		}
	}



	__global__ void FillVBO (unsigned int* pixels, int imageSize, int r, int g, int b ){
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		if (id<imageSize){
		pixels[id] = GET_RGBA(r,g,b,255);//GET_WHITE;
		}
	}



	__device__ float3 HSVtoRGB( float h , float s , float v )
	{
		float3 rgb;
		//return hsv;
		int i;
		float f, p, q, t;
		if( s == 0 ) {
			// achromatic (grey)
			rgb.x = rgb.y = rgb.z = v;
			return;
		}
		h /= 60;			// sector 0 to 5
		i = floor( h );
		f = h - i;			// factorial part of h
		p = v * ( 1 - s );
		q = v * ( 1 - s * f );
		t = v * ( 1 - s * ( 1 - f ) );
		switch( i ) {
		case 0:
			rgb.x = v;
			rgb.y = t;
			rgb.z = p;
			break;
		case 1:
			rgb.x = q;
			rgb.y = v;
			rgb.z = p;
			break;
		case 2:
			rgb.x = p;
			rgb.y = v;
			rgb.z = t;
			break;
		case 3:
			rgb.x = p;
			rgb.y = q;
			rgb.z = v;
			break;
		case 4:
			rgb.x = t;
			rgb.y= p;
			rgb.z = v;
			break;
		default:		// case 5:
			rgb.x = v;
			rgb.y = p;
			rgb.z = q;
			break;
		}
	}

    __global__ void OFMapObserver (unsigned int* pixels, int imageSize, float* of){
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

        float2 OF_value;
        float OF_size;
        float OF_angle;

        float PI = 3.14159265f;

        float h, s;

		if (id<imageSize){
            OF_value.x = of[id];
            OF_value.y = of[id+imageSize];

            OF_size  = (float) sqrt( (OF_value.x+OF_value.y) * (OF_value.x+OF_value.y) );  // normalized to be <0,1>
            OF_angle = (float) atan2(OF_value.x,OF_value.y);  // <-PI;PI>
            OF_angle = (OF_angle + PI)/2.0f;//(OF_angle+PI)/2.0f/PI;  // normalized to be <0,1>
            

            h = OF_angle;
            s = OF_size/5.0f;

            pixels[id] = hsva_to_uint_rgba(h, s, 1.0f, 1.0f);
		}
	}

    __global__ void OFConvert2AngleSize (float*of, int imageSize){
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

        float2 OF_value;
        float OF_size;
        float OF_angle;

        float PI = 3.14159265f;

		if (id<imageSize){
            OF_value.x = of[id];
            OF_value.y = of[id+imageSize];

            OF_size = (float) sqrt( (OF_value.x+OF_value.y) * (OF_value.x+OF_value.y) );  // normalized to be <0,1>
            OF_angle = (float) atan2(OF_value.x,OF_value.y);  // <-PI;PI>

            of[id] = OF_angle;
            of[id+imageSize] = OF_size;

		}
	}


}




/*
__device__ unsigned int hsva_to_uint_rgba(float h, float s, float v, float a) {
	float r, g, b;	

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

	unsigned char red = (unsigned char) (255.0f * r);
	unsigned char green = (unsigned char) (255.0f * g);
	unsigned char blue = (unsigned char) (255.0f * b);
	unsigned char alpha = (unsigned char) (255.0f * a);	

	return  (alpha << 24)  | (red << 16) | (green << 8) | blue;			
}

*/



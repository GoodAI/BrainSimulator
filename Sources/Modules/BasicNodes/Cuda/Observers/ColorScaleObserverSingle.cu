#define _SIZE_T_DEFINED 

#include <cuda.h> 
#include <math_constants.h> 
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_types.h> 
#include <vector_functions.h> 
#include <float.h>

#include "ColorHelpers.cu"

extern "C"  
{	
	//kernel code
	__global__ void ColorScaleObserverSingle(float* values, int method, int scale, float minValue, float maxValue, unsigned int* pixels, int numOfPixels)
	{		
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		if(id < numOfPixels) //id of the thread is valid
		{	
			pixels[id] = float_to_uint_rgba(values[id], method, scale, minValue, maxValue);
		}
	}

	__global__ void DrawVectorsKernel(float* values, int elements, float maxValue, unsigned int* pixels, int numOfPixels) 
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	
				+ blockDim.x*blockIdx.x				
				+ threadIdx.x;

		if(id < numOfPixels) //id of the thread is valid
		{
			float x = values[id] / maxValue;
			float y = values[numOfPixels + id] / maxValue;

			if (elements == 2) {

				float hue = atan2f(x, y) / CUDART_PI_F * 0.5f + 0.5f;				
				float value = fminf(sqrtf(x * x + y * y), 1.0f);

				pixels[id] = hsva_to_uint_rgba(hue, 1.0f, value, 1.0f);
			}
			else {
				
				float z = values[2 * numOfPixels + id] / maxValue;

				x = fminf(fmaxf(x, -1), 1);
				y = fminf(fmaxf(y, -1), 1);
				z = fminf(fmaxf(z, -1), 1);

				unsigned char red = (unsigned char) __float2uint_rn(127.5f * (x + 1));
				unsigned char green = (unsigned char) __float2uint_rn(127.5f * (y + 1));
				unsigned char blue = (unsigned char) __float2uint_rn(127.5f * (z + 1));		

				pixels[id] = (0xFF << 24) | (red << 16) | (green << 8) | blue;		
			}
		}
	}

	
	__global__ void DrawRGBKernel(float* values, unsigned int* pixels, int numOfPixels) 
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	
				+ blockDim.x*blockIdx.x				
				+ threadIdx.x;

		if(id < numOfPixels) //id of the thread is valid
		{
			float fred   = values[0 * numOfPixels + id];
			float fgreen = values[1 * numOfPixels + id];
			float fblue  = values[2 * numOfPixels + id];

			fred   = fminf(fmaxf(fred,   0), 1) * 255;
			fgreen = fminf(fmaxf(fgreen, 0), 1) * 255;
			fblue  = fminf(fmaxf(fblue,  0), 1) * 255;

			unsigned char red   = (unsigned char) __float2uint_rn(fred);
			unsigned char green = (unsigned char) __float2uint_rn(fgreen);
			unsigned char blue  = (unsigned char) __float2uint_rn(fblue);		

			pixels[id] = (0xFF << 24) | (red << 16) | (green << 8) | blue;		
		}
	}

	__global__ void DrawGrayscaleKernel(float* values, unsigned int* pixels, int numOfPixels)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		if (id < numOfPixels) //id of the thread is valid
		{
			pixels[id] = grayscale_to_uint_rgba(fminf(fmaxf(values[id], 0.0f), 1.0f));
		}
	}

}
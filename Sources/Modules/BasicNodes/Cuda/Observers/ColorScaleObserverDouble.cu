#define _SIZE_T_DEFINED 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __cplusplus 
#define __cplusplus 
#endif

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
	__global__ void ColorScaleObserverDouble(double* values, int method, int scale, float minValue, float maxValue, unsigned int* pixels, int numOfPixels)
	{		
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		if(id < numOfPixels) //id of the thread is valid
		{	
			pixels[id] = float_to_uint_rgba(values[id], method, scale, minValue, maxValue);
		}
	}

	__global__ void ColorScaleObserverTiledDouble(double* values, int method, int scale, float minValue, float maxValue, unsigned int* pixels, int numOfPixels, int tw, int th, int tilesInRow)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		if (id < numOfPixels)
		{
			// TODO custom indexing goes here!
			//pixels[id] = float_to_uint_rgba(values[id], method, scale, minValue, maxValue);
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
}
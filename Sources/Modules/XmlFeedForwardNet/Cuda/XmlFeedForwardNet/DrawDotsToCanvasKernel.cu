#define _SIZE_T_DEFINED 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __cplusplus 
#define __cplusplus 
#endif

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>


extern "C"  
{
	// 1 thread = 1 pixel

	//kernel code
	__global__ void DrawDotsToCanvasKernel(	float *src, unsigned int srcOffset, unsigned int dotNb, unsigned int dotSize, unsigned int dotMargin,
											float manualMinValue, float manualMaxValue,
											unsigned int *dest, unsigned int destWidth, unsigned int destRectX, unsigned int destRectY,
											int colorPolicy,
											float* minMax,
											unsigned int count
											)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
		
		if (id >= count)
			return;

		int dotArea = dotSize * dotSize;

		int localX = id % dotSize;
		int localY = (id % dotArea) / dotSize;

		int destPixelX = destRectX + localX;
		int destPixelY = (id / dotArea) * (dotSize + dotMargin) + localY;

		float value = src[srcOffset + id / dotArea];
		
		unsigned int color;

		if (isinf(value) && value > 0)
		{
			if (value > 0)
				color = 0xFF00FFFF; // Magenta color
			else
				color = 0xFFFF00FF; // Cyan color
		}
		else if (isnan(value))
		{
			// NaN => Blue Pixel
			color = 0xFF0000FF;
		}
		else
		{
			// Not NaN
			float minValue;
			float maxValue;
			if (minMax != 0)
			{
				minValue = minMax[0];
				maxValue = minMax[1];
			}
			else
			{
				minValue = manualMinValue;
				maxValue = manualMaxValue;
			}

			if (colorPolicy == 0) // Grey level
			{
				float ratio = (value - minValue) / (maxValue - minValue);
				if (ratio > 1)
					ratio = 1;
				else if (ratio < 0)
					ratio = 0;

				unsigned int red = 255.0 * ratio;
				unsigned int green = 255.0 * ratio;
				unsigned int blue = 255.0 * ratio;
				color = 0xFF000000 | (red << 16) | (green << 8) | blue;
			}


			else if (colorPolicy == 1) // Green and Red level
			{
				if (value == 0)
				color = 0xFF000000; 
				else if (value > 0) 
				{
					float ratio = value / maxValue;
					if (ratio > 1)
						ratio = 1.0;
					else if (ratio < 0)
						ratio = 0.0;
					unsigned int green = 255.0 * ratio;
					color = 0xFF000000 | (green << 8);
				}
				else if (value < 0)
				{
					float ratio = value / minValue;
					if (ratio > 1)
						ratio = 1.0;
					else if (ratio < 0)
						ratio = 0.0;
					unsigned int red = 255.0 * ratio;
					color = 0xFF000000 | (red << 16);
				}
			}
			else
			{
				// Add new display method here
			}
		}


		dest[destPixelX + destPixelY * destWidth] = color;
	}
}
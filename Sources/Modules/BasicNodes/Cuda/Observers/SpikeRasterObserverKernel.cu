#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

#include "ColorHelpers.cu"

//Includes for IntelliSense 
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif

extern "C"  
{
	__constant__ int D_COUNT;
	__constant__ int D_X_SIZE;
	__constant__ unsigned int D_BACKGROUND_COLOR_1;
	__constant__ unsigned int D_BACKGROUND_COLOR_2;
	__constant__ unsigned int D_MARKER_COLOR;
	__constant__ int D_GRID_STEP;
	__constant__ unsigned int D_GRID_COLOR;
	//kernel code
	__global__ void SpikeRasterObserverKernel(
		unsigned int* pixels,
		float *spikeValues,
		int offset,
		int ringArrayStart,
		int gridStepCounter,
		int renderingMethod,
		float minValue,
		float maxValue)
	{
		int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

		if(globalThreadId < D_COUNT)
		{
			int newest = ringArrayStart;
			unsigned int colorToWrite;
			if(spikeValues[globalThreadId + offset] == 0)
			{
				colorToWrite = (globalThreadId % 2 == 1) * D_BACKGROUND_COLOR_1 + (globalThreadId % 2 == 0) * D_BACKGROUND_COLOR_2;
				if(gridStepCounter == 0)
				{
					colorToWrite = D_GRID_COLOR;
				}
			}
			else
			{
				colorToWrite = float_to_uint_rgba(spikeValues[globalThreadId + offset],
					renderingMethod, 0 /*scale=Linear*/, minValue, maxValue);
			}
			pixels[globalThreadId * D_X_SIZE + newest] = colorToWrite;
			pixels[globalThreadId * D_X_SIZE + (newest + 1) % D_X_SIZE] = D_MARKER_COLOR;
		}
	}
}
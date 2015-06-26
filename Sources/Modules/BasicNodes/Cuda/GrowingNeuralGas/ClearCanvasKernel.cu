#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

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
	__constant__ unsigned int D_BACKGROUND;

	__constant__ int D_X_PIXELS;
	__constant__ int D_Y_PIXELS;

	//kernel code
	__global__ void ClearCanvasKernel(
		
		unsigned int *buffer
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < D_X_PIXELS * D_Y_PIXELS)
		{
			buffer[threadId] = D_BACKGROUND;
		}
	}

}
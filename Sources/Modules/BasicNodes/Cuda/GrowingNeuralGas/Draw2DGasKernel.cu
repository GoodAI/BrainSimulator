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
	__constant__ unsigned int D_MARKER_COLOR;
	__constant__ unsigned int D_WINNER_1_COLOR;
	__constant__ unsigned int D_WINNER_2_COLOR;

	__constant__ float D_X_MIN;
	__constant__ float D_X_MAX;
	__constant__ float D_Y_MIN;
	__constant__ float D_Y_MAX;

	__constant__ int D_X_PIXELS;
	__constant__ int D_Y_PIXELS;

	//kernel code
	__global__ void Draw2DGasKernel(
		
		int s1,
		int s2,
		float *referenceVector,
		int referenceVectorSize,
		int *activityFlag,
		int maxCells,
		unsigned int *buffer
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < maxCells)
		{
			if(activityFlag[threadId] == 1)
			{
				float xValue = referenceVector[threadId * referenceVectorSize + 0];
				float yValue = referenceVector[threadId * referenceVectorSize + 1];

				if(xValue >= D_X_MIN && xValue < D_X_MAX && yValue >= D_Y_MIN && yValue < D_Y_MAX)
				{
					float xStep = (D_X_MAX - D_X_MIN) / (float)D_X_PIXELS;
					float yStep = (D_Y_MAX - D_Y_MIN) / (float)D_Y_PIXELS;
			
					float biasedXValue = xValue - D_X_MIN;
					float biasedYValue = (D_Y_MAX - yValue);

					int xId = (int)(biasedXValue / xStep);
					int yId = (int)(biasedYValue / yStep);
					
					buffer[yId * D_X_PIXELS + xId] = 0;
					buffer[yId * D_X_PIXELS + xId] = (threadId == s1) * D_WINNER_1_COLOR + (threadId == s2) * D_WINNER_2_COLOR + (threadId != s1 && threadId != s2) * D_MARKER_COLOR;
				}
			}


			
		}
	}

}
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

	__constant__ float D_X_MIN;
	__constant__ float D_X_MAX;
	__constant__ float D_Y_MIN;
	__constant__ float D_Y_MAX;

	__constant__ int D_X_PIXELS;
	__constant__ int D_Y_PIXELS;

	//kernel code
	__global__ void Draw2DConnectionKernel(
		
		int *connections,
		float *referenceVector,
		int referenceVectorSize,
		int maxCells,
		unsigned int *buffer
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
		
		if(threadId < maxCells * maxCells)
		{
			if(connections[threadId] == 1)
			{
			int from = threadId / maxCells;
			int to = threadId % maxCells;

				if(to > from)
				{
					float AX = referenceVector[from * referenceVectorSize + 0];
					float AY = referenceVector[from * referenceVectorSize + 1];
					float BX = referenceVector[to * referenceVectorSize + 0];
					float BY = referenceVector[to * referenceVectorSize + 1];

					if(AX >= D_X_MIN && AX < D_X_MAX && AY >= D_Y_MIN && AY < D_Y_MAX && BX >= D_X_MIN && BX < D_X_MAX && BY >= D_Y_MIN && BY < D_Y_MAX)
					{
						float uX = BX - AX;
						float uY = BY - AY;

						float distance = sqrtf( uX * uX + uY * uY );

						float xStep = (D_X_MAX - D_X_MIN) / (float)D_X_PIXELS;
						float yStep = (D_Y_MAX - D_Y_MIN) / (float)D_Y_PIXELS;

						float minStep = fminf(xStep, yStep);

						float t = 0.00f;

						while(t <= 1.00f)
						{
							float actualX = AX + t * uX;
							float actualY = AY + t * uY;	

							float biasedXValue = actualX - D_X_MIN;
							float biasedYValue = (D_Y_MAX - actualY);
				
							int xId = (int)(biasedXValue / xStep);
							int yId = (int)(biasedYValue / yStep);

							buffer[yId * D_X_PIXELS + xId] = D_MARKER_COLOR;

							t = t + minStep / distance;
						}
					}
				}
			}

		}
		
	}

}
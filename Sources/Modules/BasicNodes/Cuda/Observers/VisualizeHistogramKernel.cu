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
	__constant__ int D_BINS;
	__constant__ int D_BIN_PIXEL_WIDTH;
	__constant__ int D_BIN_PIXEL_HEIGHT;

	__constant__ unsigned int D_COLOR_ONE;
	__constant__ unsigned int D_COLOR_TWO;
	__constant__ unsigned int D_COLOR_BACKGROUND;
	__constant__ unsigned int D_OUT_OF_BOUNDS;

	//kernel code
	__global__ void VisualizeHistogramKernel(

		int *globalHist,
		unsigned int* pixels

		)
	{
		int globalThreadId = blockDim.x * blockIdx.x + threadIdx.x;
		
		__shared__ int maxValue;

		if(globalThreadId < D_BINS * D_BIN_PIXEL_WIDTH)
		{
			if(threadIdx.x == 0)
			{
				maxValue = 0;
				//search maximum value for each histogram bins
				for(int h = 0; h < D_BINS; h++)
				{
					if(globalHist[h] > maxValue)
					{
						maxValue = globalHist[h];
					}
				}
			}

			__syncthreads();

			//get the height of the actual column
			int columnHeightInv = D_BIN_PIXEL_HEIGHT - (int)((double)D_BIN_PIXEL_HEIGHT * ((double)globalHist[blockIdx.x] / (double)maxValue));
			unsigned int histColor;

			if(blockIdx.x == 0 || blockIdx.x == D_BINS - 1)
			{
				histColor = D_OUT_OF_BOUNDS;
			}
			else
			{
				histColor = (blockIdx.x % 2 == 0) * D_COLOR_ONE + (blockIdx.x % 2 == 1) * D_COLOR_TWO;
			}

			for(int i = 0; i < D_BIN_PIXEL_HEIGHT; i++)
			{
				if(i < columnHeightInv)
				{
					//background color
					pixels[D_BINS * D_BIN_PIXEL_WIDTH * i + blockIdx.x*D_BIN_PIXEL_WIDTH + threadIdx.x] = D_COLOR_BACKGROUND;
				}
				else
				{
					//color of histogram
					pixels[D_BINS * D_BIN_PIXEL_WIDTH * i + blockIdx.x*D_BIN_PIXEL_WIDTH + threadIdx.x] = histColor;
				}
			}

			
		}
	}
}
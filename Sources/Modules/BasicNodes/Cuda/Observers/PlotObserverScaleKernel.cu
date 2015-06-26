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


extern "C"  
{	
	__constant__ int D_NB_CURVES;
	__constant__ int D_TEXTURE_WIDTH;
	__constant__ int D_PLOTAREA_WIDTH;
	__constant__ int D_PLOTAREA_HEIGHT;
	__constant__ int D_PLOTAREA_OFFSET_X;
	__constant__ double D_MIN_VALUE;
	__constant__ double D_MAX_VALUE;
	__constant__ unsigned int D_COLOR_BACKGROUND;
	__constant__ unsigned int D_COLOR_CURVES[6];
	__constant__ unsigned int D_COLOR_CURVE_EXTRA;

	//kernel code
	__global__ void PlotObserverScaleKernel(unsigned int* canvas, int columnStart, int nbColumn, float* values)
	{		
		int id = blockDim.x*blockIdx.y*gridDim.x	
				+ blockDim.x*blockIdx.x				
				+ threadIdx.x;

		if (id >= nbColumn * D_PLOTAREA_HEIGHT)
			return;
		
		int x = (columnStart + id % nbColumn) % D_PLOTAREA_WIDTH; 
		int y = id / nbColumn;


		double valueRange = D_MAX_VALUE - D_MIN_VALUE;
		double pixelRange = valueRange / D_PLOTAREA_HEIGHT;
		
		double displayWindowMinValue = y * pixelRange;
		double displayWindowMaxValue = displayWindowMinValue + pixelRange;


		
		// For each curve
		unsigned int color = D_COLOR_BACKGROUND;
		for (int c = 0; c < D_NB_CURVES; c++)
		{
			double currentValue = values[x * D_NB_CURVES + c];
			double previousValue = (x > 0) ? values[(x - 1) * D_NB_CURVES + c] : currentValue;
			
			double currentRelativeValue = currentValue - D_MIN_VALUE; // Use a relative value to keep the high precision if the value is very high
			double previousRelativeValue = previousValue - D_MIN_VALUE;

			if (	currentRelativeValue >= displayWindowMinValue && currentRelativeValue < displayWindowMaxValue
				|| previousRelativeValue < displayWindowMinValue && currentRelativeValue > displayWindowMaxValue
				|| previousRelativeValue > displayWindowMaxValue && currentRelativeValue < displayWindowMinValue
				)
			{
				color = (c < 6) ? D_COLOR_CURVES[c] : D_COLOR_CURVE_EXTRA;
				break;
			}
		}

		canvas[D_PLOTAREA_OFFSET_X + x + (D_PLOTAREA_HEIGHT - 1 - y) * D_TEXTURE_WIDTH] = color;	
	}
}
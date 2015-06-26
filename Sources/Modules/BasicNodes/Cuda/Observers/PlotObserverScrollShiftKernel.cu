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
	__constant__ int D_TEXTURE_WIDTH;
	__constant__ int D_PLOTAREA_WIDTH;
	__constant__ int D_PLOTAREA_HEIGHT;
	__constant__ int D_PLOTAREA_OFFSET_X;

	//kernel code
	__global__ void PlotObserverScrollShiftKernel(unsigned int* canvas)
	{		
		int id = blockDim.x*blockIdx.y*gridDim.x	
				+ blockDim.x*blockIdx.x				
				+ threadIdx.x;

		if (id >= D_PLOTAREA_HEIGHT)
			return;
		
		// This reverse coordinate are important in order for the texture 
		// to be processed in left->right order to avoid read/write conflicts
		int y = id;
		int xRange = D_PLOTAREA_WIDTH - 1;

		for (int x = 0; x < xRange; x++)
		{
			int addr = D_PLOTAREA_OFFSET_X + x + y * D_TEXTURE_WIDTH;
			canvas[addr] = canvas[addr + 1];
		}
	}
}
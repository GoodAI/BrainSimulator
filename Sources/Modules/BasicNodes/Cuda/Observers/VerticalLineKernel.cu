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


#define COLOR_GREY	0xFF888888

extern "C"  
{	

	//kernel code
	__global__ void VerticalLineKernel(unsigned int* canvas, int column, int width, int height)
	{		
		int id = blockDim.x*blockIdx.y*gridDim.x	
				+ blockDim.x*blockIdx.x				
				+ threadIdx.x;

		if (id >= height)
			return;
		
		canvas[column + id * width] = COLOR_GREY;	
	}
}
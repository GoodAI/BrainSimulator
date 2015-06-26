//Includes for IntelliSense 
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
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

extern "C"  
{	
	//kernel code
	__global__ void IntervalToBinaryVector(float input, float* outputs, int steps)
	{		
		int id = blockDim.x*blockIdx.y*gridDim.x	
				+ blockDim.x*blockIdx.x				
				+ threadIdx.x;

		if(id < steps) 
		{
			float fraction = 1.0f / steps;			
			outputs[id] = input >= fraction * id && input <= fraction * (id + 1);
		}		
	}
}
//Includes for IntelliSense 
#define _SIZE_T_DEFINED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math.h>



extern "C"  
{
	__constant__ int D_INPUT_UNITS;
	__constant__ int D_HIDDEN_UNITS;
	__constant__ int D_OUTPUT_UNITS;
	

	
	//kernel code
	__global__ void OutputDeltaKernel(float *activation, float *target, float *outputDelta)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;
		if (id < D_OUTPUT_UNITS)
		{
			outputDelta[id] = target[id] - activation[1 + D_INPUT_UNITS + D_HIDDEN_UNITS + id];
		}
	}
}
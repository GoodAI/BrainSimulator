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
	__global__ void OutputDeltaKernel(float *outputDeltas, float *target, float *outputActivations, float *outputActivationDerivatives)
	{
		int unitId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;
		if (unitId < D_OUTPUT_UNITS)
		{
			outputDeltas[unitId] = (target[unitId] - outputActivations[unitId]) * outputActivationDerivatives[unitId];
		}
	}
}
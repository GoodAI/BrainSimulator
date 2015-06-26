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
	__constant__ int D_UNITS;
	__constant__ int D_HIDDEN_UNIT_WEIGHTS;
	__constant__ int D_OUTPUT_UNIT_WEIGHTS;
	__constant__ int D_WEIGHTS;
	__constant__ float D_LEARNING_RATE;
	__constant__ float D_MOMENTUM_RATE;
	

	
	//kernel code
	__global__ void ChangeWeightsKernel(float *rtrlDerivative, float *outputDelta, float *weightsDelta, float *weights)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;
		if (id < D_WEIGHTS)
		{
			float weightDelta = 0.0f;

			for (int i = 0; i < D_OUTPUT_UNITS; i++)
			{
				weightDelta += outputDelta[i] * rtrlDerivative[id * (D_HIDDEN_UNITS + D_OUTPUT_UNITS) + D_HIDDEN_UNITS + i];
			}

			weightDelta = D_LEARNING_RATE * weightDelta + D_MOMENTUM_RATE * weightsDelta[id];
			weightsDelta[id] = weightDelta;
			weights[id] += weightDelta;
		}
	}
}

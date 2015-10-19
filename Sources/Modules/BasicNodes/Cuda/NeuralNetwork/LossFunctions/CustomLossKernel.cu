//Includes for IntelliSense 
#define _SIZE_T_DEFINED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math.h>

#include "..\Activation\ActivationFunction.cu"

extern "C"
{
	__global__ void CustomLossKernel(
		ActivationFunctionEnum actFunc,
		float *neuronInputPtr,
		float *targetPtr,
		float *deltaPtr,
		int thisLayerSize,
		int batchSize
		)
	{
		int tid = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x					//blocks preceeding current block
			+ threadIdx.x;

		if (tid < thisLayerSize * batchSize)
		{
			deltaPtr[tid] = 0;

			if (!isnan(targetPtr[tid]))
				deltaPtr[tid] = EvaluateDerivative(actFunc, neuronInputPtr[tid]) * targetPtr[tid];
		}

	}
}

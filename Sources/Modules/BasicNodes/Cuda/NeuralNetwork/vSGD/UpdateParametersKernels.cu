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
	__global__ void FullyConnectedUpdateParametersKernel(
		float *weightPtr,
		float *biasPtr,
		float *weightLearningRatePtr,
		float *biasLearningRatePtr,
		float *weightGradientPtr,
		float *biasGradientPtr,
		float *dropoutMaskPtr,
		int prevLayerSize,
		int thisLayerSize
		)
	{
		// i: prev. layer neuron id
		// j: current layer neuron id
		int i;
		int j = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (j < thisLayerSize)
		{
			if (!dropoutMaskPtr[j])
			{
				// update weights
				int index = j;
				for (i = 0; i < prevLayerSize; i++)
				{
					weightPtr[index] -= weightLearningRatePtr[index] * weightGradientPtr[index];

					index += thisLayerSize;
				}

				// update bias
				biasPtr[j] -= biasLearningRatePtr[j] * biasGradientPtr[j];
			}
		}
	}
}
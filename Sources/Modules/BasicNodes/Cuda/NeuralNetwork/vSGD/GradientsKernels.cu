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
	__device__ int sign(float val)
	{
		return (val > 0) - (val < 0);
	}

	__global__ void FullyConnectedGradientsKernel(
		float *inputPtr,
		float *deltaPtr,
		float *weightPtr,
		float L1Lambda,
		float L2Lambda,
		float *dropoutMaskPtr,
		float *weightGradPtr,
		float *biasGradPtr,
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
				// weight gradient
				int index = j;
				for (i = 0; i < prevLayerSize; i++)
				{
					weightGradPtr[index] = deltaPtr[j] * inputPtr[i] + L1Lambda * sign(weightPtr[index]) + L2Lambda * weightPtr[index];
					index += thisLayerSize;
				}

				// bias gradient
				biasGradPtr[j] = deltaPtr[j];
			}
		}
	}
}
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
	__global__ void FullyConnectedShiftKernel(
		float *weightPtr,
		float *biasPtr,
		float *shiftedWeightsPtr,
		float *shiftedBiasPtr,
		float *avgWeightGradPtr,
		float *avgBiasGradPtr,
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
				// weight gradient
				int index = j;
				for (i = 0; i < prevLayerSize; i++)
				{
					shiftedWeightsPtr[index] = weightPtr[index] + avgWeightGradPtr[index]; // TODO: Check if it is correct to add here, or if it should be subtracted
					index += thisLayerSize;
				}

				// bias gradient
				shiftedBiasPtr[j] = biasPtr[j] - avgBiasGradPtr[j];
			}
		}
	}
}
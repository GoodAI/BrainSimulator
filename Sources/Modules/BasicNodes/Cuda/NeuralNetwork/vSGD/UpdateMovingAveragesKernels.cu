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
	__global__ void FullyConnectedUpdateMovingAveragesKernel(
		float *weightsGradPtr,
		float *biasGradPtr,
		float *weightsGradCurvePtr,
		float *biasGradCurvePtr,
		float *avgWeightGradPtr,
		float *avgBiasGradPtr,
		float *avgWeightGradVarPtr,
		float *avgBiasGradVarPtr,
		float *avgWeightGradCurvePtr,
		float *avgBiasGradCurvePtr,
		float *avgWeightGradCurveVarPtr,
		float *avgBiasGradCurveVarPtr,
		float *weightMemorySizePtr,
		float *biasMemorySizePtr,
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
				int index = j;
				for (i = 0; i < prevLayerSize; i++)
				{
					// update moving averages according to memory size
					avgWeightGradPtr[index] = (1.0f - 1.0f / weightMemorySizePtr[index]) * avgWeightGradPtr[index] + (1.0f / weightMemorySizePtr[index]) * weightsGradPtr[index];
					avgWeightGradVarPtr[index] = (1.0f - 1.0f / weightMemorySizePtr[index]) * avgWeightGradVarPtr[index] + (1.0f / weightMemorySizePtr[index]) * weightsGradPtr[index] * weightsGradPtr[index];
					avgWeightGradCurvePtr[index] = (1.0f - 1.0f / weightMemorySizePtr[index]) * avgWeightGradCurvePtr[index] + (1.0f / weightMemorySizePtr[index]) * weightsGradCurvePtr[index];
					avgWeightGradCurveVarPtr[index] = (1.0f - 1.0f / weightMemorySizePtr[index]) * avgWeightGradCurveVarPtr[index] + (1.0f / weightMemorySizePtr[index]) * weightsGradCurvePtr[index] * weightsGradCurvePtr[index];

					index += thisLayerSize;
				}

				// update moving averages according to memory size
				avgBiasGradPtr[j] = (1.0f - 1.0f / biasMemorySizePtr[j]) * avgBiasGradPtr[j] + (1.0f / biasMemorySizePtr[j]) * biasGradPtr[j];
				avgBiasGradVarPtr[j] = (1.0f - 1.0f / biasMemorySizePtr[j]) * avgBiasGradVarPtr[j] + (1.0f / biasMemorySizePtr[j]) * biasGradPtr[j] * biasGradPtr[j];
				avgBiasGradCurvePtr[j] = (1.0f - 1.0f / biasMemorySizePtr[j]) * avgBiasGradCurvePtr[j] + (1.0f / biasMemorySizePtr[j]) * biasGradCurvePtr[j];
				avgBiasGradCurveVarPtr[j] = (1.0f - 1.0f / biasMemorySizePtr[j]) * avgBiasGradCurveVarPtr[j] + (1.0f / biasMemorySizePtr[j]) * biasGradCurvePtr[j] * biasGradCurvePtr[j];
			}
		}
	}
}
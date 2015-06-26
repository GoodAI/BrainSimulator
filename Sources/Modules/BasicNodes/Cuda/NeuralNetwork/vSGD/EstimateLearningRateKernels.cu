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
	__global__ void FullyConnectedEstimateLearningRateKernel(
		float *weightLearningRatePtr,
		float *biasLearningRatePtr,
		float *avgWeightGradPtr,
		float *avgBiasGradPtr,
		float *avgWeightGradVarPtr,
		float *avgBiasGradVarPtr,
		float *avgWeightGradCurvePtr,
		float *avgBiasGradCurvePtr,
		float *avgWeightGradCurveVarPtr,
		float *avgBiasGradCurveVarPtr,
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
					// estimate the learning rate
					weightLearningRatePtr[index] = (avgWeightGradCurvePtr[index] / avgWeightGradCurveVarPtr[index]) * (avgWeightGradPtr[index] * avgWeightGradPtr[index] / avgWeightGradVarPtr[index]);

					index += thisLayerSize;
				}

				// estimate the learning rate
				biasLearningRatePtr[j] = (avgBiasGradCurvePtr[j] / avgBiasGradCurveVarPtr[j]) * (avgBiasGradPtr[j] * avgBiasGradPtr[j] / avgBiasGradVarPtr[j]);
			}
		}
	}
}
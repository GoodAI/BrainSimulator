//Includes for IntelliSense 
#define _SIZE_T_DEFINED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math.h>

#include "../Activation/ActivationFunction.cu"

extern "C"
{
	__global__ void FullyConnectedSGDUpdateKernel(
		float *inputPtr,
		float *deltaPtr,
		float *weightPtr,
		float *previousWeightDeltaPtr,
		float *biasPtr,
		float *previousBiasDeltaPtr,
		float trainingRate,
		float momentum,
		float L1Lambda,
		float L2Lambda,
		float *dropoutMaskPtr,
		int prevLayerSize,
		int thisLayerSize
		)
	{
		// i: prev. layer neuron id
		// j: current layer neuron id
		float weightDelta;
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
					//weightDelta = trainingRate * deltaPtr[j] * inputPtr[i];
					weightDelta = trainingRate * (deltaPtr[j] * inputPtr[i] + L1Lambda * sign(weightPtr[index]) + L2Lambda * weightPtr[index]);
					if (momentum != 0)
					{
						weightDelta += momentum * previousWeightDeltaPtr[index];
						previousWeightDeltaPtr[index] = weightDelta;
					}

					weightPtr[index] -= weightDelta;

					index += thisLayerSize;
				}

				// update bias
				float biasDelta = trainingRate * deltaPtr[j];
				if (momentum != 0)
				{
					biasPtr[j] -= momentum * previousBiasDeltaPtr[j];
					previousBiasDeltaPtr[j] = biasDelta;
				}
				biasPtr[j] -= biasDelta;
			}
		}
	}

	__global__ void FullyConnectedRMSPropUpdateKernel(
		float *inputPtr,
		float *deltaPtr,
		float *weightPtr,
		float *previousWeightDeltaPtr,
		float *biasPtr,
		float *previousBiasDeltaPtr,
		float trainingRate,
		float momentum,
		float L1Lambda,
		float L2Lambda,
		float *dropoutMaskPtr,
		int prevLayerSize,
		int thisLayerSize,
		float *meanSquareWeight,
		float *meanSquareBias,
		float smoothingFactor
		)
	{
		// i: prev. layer neuron id
		// j: current layer neuron id
		float gradient;
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
					//weightDelta = trainingRate * deltaPtr[j] * inputPtr[i];
					gradient = deltaPtr[j] * inputPtr[i] + L1Lambda * sign(weightPtr[index]) + L2Lambda * weightPtr[index];
					if (momentum != 0)
					{
						gradient += momentum * previousWeightDeltaPtr[index];
						previousWeightDeltaPtr[index] = gradient;
					}

					// calculate meansquare
					meanSquareWeight[index] = smoothingFactor * meanSquareWeight[index] + (1.0f - smoothingFactor) * gradient * gradient;
					if (meanSquareWeight[index] != 0)
						gradient /= sqrtf(meanSquareWeight[index]);

					weightPtr[index] -= trainingRate * gradient;

					index += thisLayerSize;
				}

				// update bias
				gradient = deltaPtr[j];
				if (momentum != 0)
				{
					biasPtr[j] -= momentum * previousBiasDeltaPtr[j];
					previousBiasDeltaPtr[j] = gradient;
				}
				// calculate meansquare
				meanSquareBias[j] = smoothingFactor * meanSquareBias[j] + (1.0f - smoothingFactor) * gradient * gradient;
				if (meanSquareBias[j] != 0)
					gradient /= sqrtf(meanSquareBias[j]);

				biasPtr[j] -= trainingRate * gradient;
			}
		}
	}
}
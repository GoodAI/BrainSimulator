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
		float *weightGradientPtr,
		float *biasGradientPtr,
		float *weightPtr,
		float *previousWeightDeltaPtr,
		float *biasPtr,
		float *previousBiasDeltaPtr,
		float trainingRate,
		float momentum,
		float L1Lambda,
		float L2Lambda,
		float *dropoutMaskPtr,
		int thisLayerSize,
		int batchSize,
		int weightCount
		)
	{
		// i: prev. layer neuron id
		// j: current layer neuron id
		int weightIdx = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
				+ blockDim.x * blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if (weightIdx < weightCount)
		{
			int j = weightIdx % thisLayerSize; // index of output neuron
			if (!dropoutMaskPtr[j])
			{
				// update weights
				int i = weightIdx / thisLayerSize; // index of input neuron

				//weightDelta = trainingRate * deltaPtr[j] * inputPtr[i];
				float weightDelta;
				if (batchSize == 1) // multiply delta and input here, because weightGradientPtr is not filled by cuBLAS
					weightDelta = trainingRate * (deltaPtr[j] * inputPtr[i] + L1Lambda * sign(weightPtr[weightIdx]) + L2Lambda * weightPtr[weightIdx]);
				else
					weightDelta = trainingRate * (weightGradientPtr[weightIdx] + L1Lambda * sign(weightPtr[weightIdx]) + L2Lambda * weightPtr[weightIdx]) / batchSize;

				if (momentum != 0)
				{
					weightDelta += momentum * previousWeightDeltaPtr[weightIdx];
					previousWeightDeltaPtr[weightIdx] = weightDelta;
				}

				weightPtr[weightIdx] -= weightDelta;

				// update bias
				if (weightIdx / thisLayerSize == 0) {
					float biasDelta;
					if (batchSize == 1) // biasGradientPtr is not filled by cuBLAS
						biasDelta = trainingRate * deltaPtr[j];
					else
						biasDelta = trainingRate * biasGradientPtr[j] / batchSize;

					if (momentum != 0)
					{
						biasDelta += momentum * previousBiasDeltaPtr[j];
						previousBiasDeltaPtr[j] = biasDelta;
					}
					biasPtr[j] -= biasDelta;
				}
			}
		}
	}

	__global__ void FullyConnectedRMSPropUpdateKernel(
		float *inputPtr,
		float *deltaPtr,
		float *weightGradientPtr,
		float *biasGradientPtr,
		float *weightPtr,
		float *previousWeightDeltaPtr,
		float *biasPtr,
		float *previousBiasDeltaPtr,
		float trainingRate,
		float momentum,
		float L1Lambda,
		float L2Lambda,
		float *dropoutMaskPtr,
		int thisLayerSize,
		int batchSize,
		int weightCount,
		float *meanSquareWeight,
		float *meanSquareBias,
		float smoothingFactor
		)
	{
		// i: prev. layer neuron id
		// j: current layer neuron id
		int weightIdx = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (weightIdx < weightCount)
		{
			int j = weightIdx % thisLayerSize; // index of output neuron
			if (!dropoutMaskPtr[j])
			{
				// update weights
				int i = weightIdx / thisLayerSize; // index of input neuron

				float gradient;
				if (batchSize == 1) // multiply delta and input here, because weightGradientPtr is not filled by cuBLAS
					gradient = (deltaPtr[j] * inputPtr[i] + L1Lambda * sign(weightPtr[weightIdx]) + L2Lambda * weightPtr[weightIdx]);
				else
					gradient = (weightGradientPtr[weightIdx] + L1Lambda * sign(weightPtr[weightIdx]) + L2Lambda * weightPtr[weightIdx]) / batchSize;

				if (momentum != 0)
				{
					gradient += momentum * previousWeightDeltaPtr[weightIdx];
					previousWeightDeltaPtr[weightIdx] = gradient;
				}

				// calculate meansquare
				meanSquareWeight[weightIdx] = smoothingFactor * meanSquareWeight[weightIdx] + (1.0f - smoothingFactor) * gradient * gradient;
				if (meanSquareWeight[weightIdx] != 0)
					gradient /= sqrtf(meanSquareWeight[weightIdx]);

				weightPtr[weightIdx] -= trainingRate * gradient;

				// update bias
				if (weightIdx / thisLayerSize == 0)
				{
					if (batchSize == 1) // biasGradientPtr is not filled by cuBLAS
						gradient = deltaPtr[j];
					else
						gradient = biasGradientPtr[j] / batchSize;

					if (momentum != 0)
					{
						gradient += momentum * previousBiasDeltaPtr[j];
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


	__global__ void FullyConnectedAdadeltaUpdateKernel(
		float *inputPtr,
		float *deltaPtr,
		float *weightGradientPtr,
		float *biasGradientPtr,
		float *weightPtr,
		float *biasPtr,
		float L1Lambda,
		float L2Lambda,
		float *dropoutMaskPtr,
		int thisLayerSize,
		int batchSize,
		int weightCount,
		float *adaSquares, float *adaDeltas, float *adaBiasSquares, float *adaBiasDeltas,
		float ro, float epsilon
		)
	{
		// i: prev. layer neuron id
		// j: current layer neuron id
		int weightIdx = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (weightIdx < weightCount)
		{
			int j = weightIdx % thisLayerSize; // index of output neuron
			if (!dropoutMaskPtr[j])
			{
				// update weights
				int i = weightIdx / thisLayerSize; // index of input neuron
				
				float gradient;
				if (batchSize == 1) // multiply delta and input here, because weightGradientPtr is not filled by cuBLAS
					gradient = (deltaPtr[j] * inputPtr[i] + L1Lambda * sign(weightPtr[weightIdx]) + L2Lambda * weightPtr[weightIdx]);
				else
					gradient = (weightGradientPtr[weightIdx] + L1Lambda * sign(weightPtr[weightIdx]) + L2Lambda * weightPtr[weightIdx]) / batchSize;

				adaSquares[weightIdx] = ro * adaSquares[weightIdx] + (1 - ro) * gradient * gradient;
				float dx = -sqrtf((adaDeltas[weightIdx] + epsilon) / (adaSquares[weightIdx] + epsilon)) * gradient;
				adaDeltas[weightIdx] = ro * adaDeltas[weightIdx] + (1 - ro) * dx * dx;
				weightPtr[weightIdx] += dx;

				// update bias
				if (weightIdx / thisLayerSize == 0)
				{
					if (batchSize == 1) // biasGradientPtr is not filled by cuBLAS
						gradient = deltaPtr[j];
					else
						gradient = biasGradientPtr[j] / batchSize;

					adaBiasSquares[j] = ro * adaBiasSquares[j] + (1 - ro) * gradient * gradient;
					float dx = -sqrtf((adaBiasDeltas[j] + epsilon) / (adaBiasSquares[j] + epsilon)) * gradient;
					adaBiasDeltas[j] = ro * adaBiasDeltas[j] + (1 - ro) * dx * dx;
					biasPtr[j] += dx;
				}
			}
		}
	}

	__global__ void PartialSGDUpdateKernel(
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
		int thisLayerSize,
		int weightCount,
		int suppressUpdatesAt,
		int suppressUpdatesCount
		)
	{
		// i: prev. layer neuron id
		// j: current layer neuron id
		int weightIdx = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (weightIdx < weightCount)
		{
			int j = weightIdx % thisLayerSize; // index of output neuron

			if (!dropoutMaskPtr[j] && (suppressUpdatesCount == 0 || j < suppressUpdatesAt || j >= suppressUpdatesAt + suppressUpdatesCount))
			{
				// update weights
				int i = weightIdx / thisLayerSize; // index of input neuron

				//weightDelta = trainingRate * deltaPtr[j] * inputPtr[i];
				float weightDelta = trainingRate * (deltaPtr[j] * inputPtr[i] + L1Lambda * sign(weightPtr[weightIdx]) + L2Lambda * weightPtr[weightIdx]);
				if (momentum != 0)
				{
					weightDelta += momentum * previousWeightDeltaPtr[weightIdx];
					previousWeightDeltaPtr[weightIdx] = weightDelta;
				}

				weightPtr[weightIdx] -= weightDelta;

				// update bias
				if (weightIdx / thisLayerSize == 0) {
					float biasDelta = trainingRate * deltaPtr[j];
					if (momentum != 0)
					{
						biasDelta += momentum * previousBiasDeltaPtr[j];
						previousBiasDeltaPtr[j] = biasDelta;
					}
					biasPtr[j] -= biasDelta;
				}
			}
		}
	}

}
//Includes for IntelliSense 
#define _SIZE_T_DEFINED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math.h>
#include <cfloat>

#include "..\Activation\ActivationFunction.cu"

extern "C"
{
	__global__ void FullyConnectedForwardKernel(
		ActivationFunctionEnum activationFunction,
		float *inputPtr,
		float *outputPtr,
		float *weightPtr,
		float *neuronInputPtr,
		float *biasPtr,
		float *dropoutMaskPtr,
		float dropout,
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
			if (dropoutMaskPtr[j])
			{
				neuronInputPtr[j] = 0;
				outputPtr[j] = 0;
			}
			else
			{
				float sum = 0.0;
				int index = j;
				for (i = 0; i < prevLayerSize; i++) {
					sum += weightPtr[index] * inputPtr[i];
					index += thisLayerSize;
				}
				// add bias
				sum += biasPtr[j];

				// sum neuron input
				neuronInputPtr[j] = sum;

				// set output value
				outputPtr[j] = Evaluate(activationFunction, sum) / (1.0f - dropout);
			}
		}
	}


	__global__ void FullyConnectedForwardBatchKernel(
		ActivationFunctionEnum activationFunction,
		float *outputPtr,
		float *neuronInputPtr,
		float *biasPtr,
		float *dropoutMaskPtr,
		float dropout,
		int thisLayerSize,
		int batchSize
		)
	{
		int threadId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;
		int neuronId = threadId % thisLayerSize;

		if (threadId < thisLayerSize * batchSize)
		{
			// add bias to neuron input and apply dropout mask
			neuronInputPtr[threadId] = !dropoutMaskPtr[neuronId] * (neuronInputPtr[threadId] + biasPtr[neuronId]);

			// set output value
			outputPtr[threadId] = !dropoutMaskPtr[neuronId] * (Evaluate(activationFunction, neuronInputPtr[threadId]) / (1.0f - dropout));
		}
	}


	__global__ void OneToOneForwardKernel(
		ActivationFunctionEnum activationFunction,
		float *inputPtr,
		float *outputPtr,
		int layerSize
		)
	{
		// i: neuron id
		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (i < layerSize)
			outputPtr[i] = Evaluate(activationFunction, inputPtr[i]);
	}


	__global__ void GaussianForwardSamplingKernel(
		ActivationFunctionEnum activationFunction,
		float* means,
		float* sigmas,
		float* noisyInput,
		float* outputPtr,
		float* randomNormalPtr,
		int prevLayerSize,
		int thisLayerSize
		)
	{
		// j: current layer neuron id
		int j = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (j < thisLayerSize)
		{
			float mu = means[j], sigma = sigmas[j], x = randomNormalPtr[j];

			// input means after applying noise
			noisyInput[j] = mu + x * powf(sigma, 2);

			// squashing function applied to noisy input
			outputPtr[j] = Evaluate(activationFunction, noisyInput[j]);
		}
	}

	__global__ void GaussianMinMaxField(float* input, int inputCount, float* mins, float* maxes)
	{
		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (i < inputCount)
		{
			mins[i] = fminf(mins[i], input[i]);
			maxes[i] = fmaxf(maxes[i], input[i]);
		}
	}

	__global__ void GaussianResetPriorStats(int inputCount, float* mins, float* maxes)
	{
		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (i < inputCount)
		{
			mins[i] = FLT_MAX;
			maxes[i] = FLT_MIN;
		}
	}

	__global__ void GaussianSamplePrior(float* input, int inputCount, float* mins, float* maxes, float* randomUniform)
	{
		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (i < inputCount)
		{
			float diff = maxes[i] - mins[i];
			input[i] = randomUniform[i] * diff + mins[i];
		}
	}

	__global__ void NegativeCorrelationForwardResetKernel(
		float* outputPtr,
		int thisLayerSize
		)
	{
		// j: current layer neuron id
		int j = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (j < thisLayerSize)
		{
			outputPtr[j] = 0;
		}
	}

	__global__ void NegativeCorrelationForwardSumKernel(
		float* inputPtr,
		float* outputPtr,
		int thisLayerSize
		)
	{
		// j: current layer neuron id
		int j = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (j < thisLayerSize)
		{
			outputPtr[j] += inputPtr[j];
		}
	}

	__global__ void NegativeCorrelationForwardDivideKernel(
		float* outputPtr,
		int thisLayerSize,
		int inputModelCount
		)
	{
		// j: current layer neuron id
		int j = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (j < thisLayerSize)
		{
			outputPtr[j] /= (float)inputModelCount;
		}
	}
}

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

// Negative correlation hyperparameter
__constant__ float Lambda;

extern "C"
{
	__global__ void FullyConnectedDeltaKernel(
		ActivationFunctionEnum prevActFunc,
		float *prevWeighedInputPtr,
		float *prevDeltaPtr,
		float *thisDeltaPtr,
		float *weightPtr,
		float dropout,
		int prevLayerSize,
		int thisLayerSize
		)
	{
		// i: prev layer neuron id
		// j: this layer neuron id
		int i = blockDim.x * blockIdx.y * gridDim.x			//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		int j;

		if (i < prevLayerSize)
		{
			float sum = 0.0;
			int index = i * thisLayerSize;
			for (j = 0; j < thisLayerSize; j++)
				sum += weightPtr[index + j] * thisDeltaPtr[j];

			sum /= 1.0f - dropout;
			sum *= EvaluateDerivative(prevActFunc, prevWeighedInputPtr[i]);
			prevDeltaPtr[i] += sum; // batch learning, remember to initialize delta
		}
	}

	__global__ void FullyConnectedDeltaBatchKernel(
		ActivationFunctionEnum prevActFunc,
		float *prevWeighedInputPtr,
		float *prevDeltaPtr,
		float dropout,
		int prevLayerSize,
		int batchSize
		)
	{
		int threadId = blockDim.x * blockIdx.y * gridDim.x			//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (threadId < batchSize * prevLayerSize)
		{
			float sum = prevDeltaPtr[threadId];

			sum /= 1.0f - dropout;
			sum *= EvaluateDerivative(prevActFunc, prevWeighedInputPtr[threadId]);
			prevDeltaPtr[threadId] = sum;
		}
	}

	__global__ void OneToOneDeltaKernel(
		ActivationFunctionEnum prevActFunc,
		float *prevWeighedInputPtr,
		float *prevDeltaPtr,
		float *thisDeltaPtr,
		int layerSize
		)
	{
		// i: neuron id
		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (i < layerSize)
			prevDeltaPtr[i] += thisDeltaPtr[i] * EvaluateDerivative(prevActFunc, prevWeighedInputPtr[i]); // batch learning, remember to initialize delta
	}


	__global__ void GaussianSamplingDeltaKernel(
		int useSigmaConstant,
		ActivationFunctionEnum prevActFunc,
		float* prevWeighedInputPtr,
		float* sigmas,
		float* meanDeltas,
		float* sigmaDeltas,
		float* thisDeltaPtr,
		float* randomNormalPtr,
		int thisLayerSize
		)
	{
		// i: prev layer neuron id
		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (i < thisLayerSize)
		{
			// no extra term from transformation when taking derivative w.r.t mean: mean + randomNormal * sigma
			meanDeltas[i] += thisDeltaPtr[i] * EvaluateDerivative(prevActFunc, prevWeighedInputPtr[i]);

			// if not using constant sigmas, then they are in second half
			// randomNormal term is because there is one more transformation before squasing: mean + randomNormal * sigma
			sigmaDeltas[i] += !useSigmaConstant * 2 * sigmas[i] * (thisDeltaPtr[i] * randomNormalPtr[i] * EvaluateDerivative(prevActFunc, prevWeighedInputPtr[i]));
		}
	}

	__global__ void NegativeCorrelationDeltaKernel(
		ActivationFunctionEnum prevActFunc,
		float* prevNeuronInput,
		float* modelOutputPtr,
		float* ensembleOutputPtr,
		int thisLayerSize,
		float* prevDeltaPtr,
		float* thisDeltaPtr,
		int inputModelCount
		)
	{
		// i: prev layer neuron id
		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (i < thisLayerSize)
		{
			thisDeltaPtr[i] -= EvaluateDerivative(prevActFunc, prevNeuronInput[i]) * Lambda * (modelOutputPtr[i] - ensembleOutputPtr[i]);
			prevDeltaPtr[i] -= EvaluateDerivative(prevActFunc, prevNeuronInput[i]) * Lambda * (modelOutputPtr[i] - ensembleOutputPtr[i]);
		}
	}

	__global__ void PreActivationFunctionDeltaKernel(
		ActivationFunctionEnum actFunc,
		float *inputPtr,
		float *deltaPtr,
		int layerSize,
		int batchSize
		)
	{
		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (i < layerSize * batchSize)
			deltaPtr[i] *= EvaluateDerivative(actFunc, inputPtr[i]);
	}
}

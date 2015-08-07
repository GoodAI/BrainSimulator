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
		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
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
		float* inputPtr,
		float* prevDeltaPtr,
		float* thisDeltaPtr,
		float* randomUniformPtr,
		int prevLayerSize,
		int thisLayerSize
		)
	{
		// i: prev layer neuron id
		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		// j: this layer neuron id
		int j = i % thisLayerSize;

		float mu = inputPtr[j];
		float sigma = inputPtr[j + thisLayerSize];

		float diff = mu - sigma;
		float diff_sq = pow(diff, 2);

		float sigma_sq = pow(sigma, 2);
		float sigma_cu = sigma_sq * sigma;

		float stdScore_sq = diff_sq / sigma_sq;

		if (i < prevLayerSize / 2)
		{
			prevDeltaPtr[i] = 2 * diff * expf(-stdScore_sq) / sigma_sq;
		}
		else if (i < prevLayerSize)
		{
			prevDeltaPtr[i] = 2 * diff_sq * expf(-stdScore_sq) / sigma_cu;
		}
	}
}
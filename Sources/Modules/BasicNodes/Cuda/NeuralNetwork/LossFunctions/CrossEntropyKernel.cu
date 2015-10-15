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
	__global__ void CrossEntropyKernel(
		ActivationFunctionEnum actFunc,
		float *neuronInputPtr,
		float *outputPtr,
		float *targetPtr,
		float *deltaPtr,
		float *costPtr,
		int thisLayerSize,
		int batchSize
		)
	{
		extern __shared__ float loss[];

		unsigned int blockSize = blockDim.x;
		unsigned int tid = threadIdx.x;
		unsigned int k = tid;

		loss[tid] = 0;
		__syncthreads();

		while (k < thisLayerSize * batchSize)
		{
			if (!isnan(targetPtr[k]))
			{
				// accumulate loss
				if (targetPtr[k])
					loss[tid] -= logf(outputPtr[k]) / batchSize; // exp(output[k])/sum of exps of outputs should be here by this time (use softmax activation)

				// calculate delta, assuming that target is a vector of 0s with exactly one 1
				deltaPtr[k] += (outputPtr[k] - targetPtr[k]);

				// meaning this would be equivalent
				// deltaPtr[k] += outputPtr[k] - (targetPtr[k] == 1);



			}
			k += blockSize;
		}

		// reduction of loss to cost
		if (blockSize >= 1024) { if (tid < 512) { loss[tid] += loss[tid + 512]; } __syncthreads(); }
		if (blockSize >= 512) { if (tid < 256) { loss[tid] += loss[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { loss[tid] += loss[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { loss[tid] += loss[tid + 64]; } __syncthreads(); }
		if (tid < 32) {
			if (blockSize >= 64) loss[tid] += loss[tid + 32];
			if (blockSize >= 32) loss[tid] += loss[tid + 16];
			if (blockSize >= 16) loss[tid] += loss[tid + 8];
			if (blockSize >= 8) loss[tid] += loss[tid + 4];
			if (blockSize >= 4) loss[tid] += loss[tid + 2];
			if (blockSize >= 2) loss[tid] += loss[tid + 1];
		}
		if (tid == 0)
			*costPtr = loss[0];
	}
}
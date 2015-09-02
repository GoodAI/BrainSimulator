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
	//
	// Loss function for comparing images.
	// See 
	// https://github.com/skaae/lasagne-draw/blob/master/deepmodels/layers/draw.py
	// line 613 - 632
	// The sigmoid function is applied to the output (to ensure range 0..1) 
	// then we take the cross entropy between target and output.
	// 

	__global__ void ImageLossKernel(
		float *canvasPtr,
		float *targetPtr,
		float *deltaPtr,
		float *costPtr,
		int thisLayerSize,
		float imageLossLearningRate
		)
	{
		extern __shared__ float loss[];

		unsigned int blockSize = blockDim.x;
		unsigned int tid = threadIdx.x;
		unsigned int k = tid;

		loss[tid] = 0;

		while (k < thisLayerSize)
		{
			if (!isnan(targetPtr[k]))
			{
				float o = sigmoid(canvasPtr[k]);
				float t = targetPtr[k];

				// Use cross entropy for the loss
				loss[tid] -= t * logf(o) + (1 - t) * logf(1 - o);

				deltaPtr[k] += imageLossLearningRate * (o - t);
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
		// Add, not assign, because we may be using this loss measure in combination with another
		if (tid == 0)
			*costPtr += loss[0];
	}
}
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
	__global__ void L1TermKernel(
		float *weightPtr,
		float *L1TermPtr,
		int weights
		)
	{
		extern __shared__ float partialSum[];

		unsigned int blockSize = blockDim.x;
		unsigned int tid = threadIdx.x;
		unsigned int idx = tid;

		partialSum[tid] = 0;
		while (idx < weights) { partialSum[tid] += abs(weightPtr[idx]); idx += blockSize; }

		if (blockSize >= 1024) { if (tid < 512) { partialSum[tid] += partialSum[tid + 512]; } __syncthreads(); }
		if (blockSize >= 512) { if (tid < 256) { partialSum[tid] += partialSum[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { partialSum[tid] += partialSum[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { partialSum[tid] += partialSum[tid + 64]; } __syncthreads(); }
		if (tid < 32) {
			if (blockSize >= 64) partialSum[tid] += partialSum[tid + 32];
			if (blockSize >= 32) partialSum[tid] += partialSum[tid + 16];
			if (blockSize >= 16) partialSum[tid] += partialSum[tid + 8];
			if (blockSize >= 8) partialSum[tid] += partialSum[tid + 4];
			if (blockSize >= 4) partialSum[tid] += partialSum[tid + 2];
			if (blockSize >= 2) partialSum[tid] += partialSum[tid + 1];
		}
		if (tid == 0)
			*L1TermPtr = partialSum[0];
	}

	__global__ void L2TermKernel(
		float *weightPtr,
		float *L2TermPtr,
		int weights
		)
	{
		extern __shared__ float partialSum[];

		unsigned int blockSize = blockDim.x;
		unsigned int tid = threadIdx.x;
		unsigned int idx = tid;

		partialSum[tid] = 0;
		while (idx < weights) { partialSum[tid] += weightPtr[idx] * weightPtr[idx]; idx += blockSize; }

		if (blockSize >= 1024) { if (tid < 512) { partialSum[tid] += partialSum[tid + 512]; } __syncthreads(); }
		if (blockSize >= 512) { if (tid < 256) { partialSum[tid] += partialSum[tid + 256]; } __syncthreads(); }
		if (blockSize >= 256) { if (tid < 128) { partialSum[tid] += partialSum[tid + 128]; } __syncthreads(); }
		if (blockSize >= 128) { if (tid < 64) { partialSum[tid] += partialSum[tid + 64]; } __syncthreads(); }
		if (tid < 32) {
			if (blockSize >= 64) partialSum[tid] += partialSum[tid + 32];
			if (blockSize >= 32) partialSum[tid] += partialSum[tid + 16];
			if (blockSize >= 16) partialSum[tid] += partialSum[tid + 8];
			if (blockSize >= 8) partialSum[tid] += partialSum[tid + 4];
			if (blockSize >= 4) partialSum[tid] += partialSum[tid + 2];
			if (blockSize >= 2) partialSum[tid] += partialSum[tid + 1];
		}
		if (tid == 0)
			*L2TermPtr = 0.5f * partialSum[0];
	}

	__global__ void DropoutMaskKernel(
		float *dropoutMaskPtr,
		float dropout,
		int inputSize
		)
	{
		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (i < inputSize)
		{
			dropoutMaskPtr[i] = dropout > dropoutMaskPtr[i];
			/*if (dropoutMaskPtr[i] > dropout)
				dropoutMaskPtr[i] = 0.0f;
			else
				dropoutMaskPtr[i] = 1.0f;*/
		}
	}
}
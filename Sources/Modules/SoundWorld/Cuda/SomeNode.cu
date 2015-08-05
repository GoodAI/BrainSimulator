
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


extern "C"
{
	__global__ void IncrementAll(float* input, float* output, float incrementSize, int itemCount)
	{
		int threadId = blockIdx.y*blockDim.x*gridDim.x
			+ blockIdx.x*blockDim.x
			+ threadIdx.x;

		if (threadId < itemCount)
		{
			output[threadId] = input[threadId] + incrementSize;
		}
	}
}
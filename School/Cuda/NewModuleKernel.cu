
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


extern "C"
{
	__global__ void NewModuleWithSourceDepsKernel(float* input, float* output, float incrementSize, int itemCount)
	{
		int threadId = blockIdx.y*blockDim.x*gridDim.x
			+ blockIdx.x*blockDim.x
			+ threadIdx.x;
	}
}
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
	__global__ void CumulateThroughTimeKernel(float* memoryBlocks, int count, int sequenceLength)
	{
		int memoryIdx = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		if (memoryIdx < count)
		{
			for (size_t i = 1; i < sequenceLength; i++)
			{
				int memoryBlockOffset = i * count;
				memoryBlocks[memoryIdx] += memoryBlocks[memoryBlockOffset + memoryIdx];
			}
		}
	}

	__global__ void CopyThroughTimeKernel(float* memoryBlocks, int count, int sequenceLength)
	{
		int memoryIdx = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		if (memoryIdx < count)
		{
			int lastMemoryBlockOffset = sequenceLength - 1;

			for (size_t i = sequenceLength - 2; i >= 0; i--)
			{
				int memoryBlockOffset = i * count;
				memoryBlocks[memoryBlockOffset + memoryIdx] = memoryBlocks[lastMemoryBlockOffset + memoryIdx];
			}
		}
	}
}

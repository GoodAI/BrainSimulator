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
		int memoryIdx = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		// 0 is boundary block
		int firstMemoryBlockId = 1;
		int firstMemoryBlockOffset = firstMemoryBlockId * count;

		// sequenceLength - 1 is boundary block
		int lastMemoryBlockId = sequenceLength - 2;

		if (memoryIdx < count)
		{
			// firstMemoryBlockId + 1 excludes the first block (which we are summing into)
			for (size_t i = firstMemoryBlockId + 1; i < lastMemoryBlockId; i++)
			{
				int memoryBlockOffset = i * count;
				memoryBlocks[firstMemoryBlockOffset + memoryIdx] += memoryBlocks[memoryBlockOffset + memoryIdx];
			}
		}
	}

	__global__ void CopyThroughTimeKernel(float* memoryBlocks, int count, int sequenceLength)
	{
		int memoryIdx = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		// 0 is boundary block
		int firstMemoryBlockId = 1;

		// sequenceLength - 1 is boundary block
		int lastMemoryBlockId = sequenceLength - 2;
		int lastMemoryBlockOffset = lastMemoryBlockId * count;

		if (memoryIdx < count)
		{
			// lastMemoryBlockId - 1 excludes last memory block (we are taking values from that block)
			for (size_t i = lastMemoryBlockId - 1; i >= firstMemoryBlockId; i--)
			{
				int memoryBlockOffset = i * count;
				memoryBlocks[memoryBlockOffset + memoryIdx] = memoryBlocks[lastMemoryBlockOffset + memoryIdx];
			}
		}
	}
}

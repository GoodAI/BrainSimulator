#define _SIZE_T_DEFINED 

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>

extern "C" 
{	

	__global__ void NormalizePositionKernel(
		float *input,
		float *normalized,
		float xMax,
		float yMax
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;		

		if(threadId < 1)
		{
			normalized[0] = input[0] / xMax;
			normalized[1] = input[1] / yMax;
		}
	}

	__global__ void InterpolateSymbolsKernel(
		float *symbolVectors,
		int symbolOneId,
		int symbolTwoId,
		float weightOne,
		float weightTwo,
		float *resultSymbol,
		int symbolSize
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < symbolSize)
		{
			int symbolOneCellId = symbolOneId * symbolSize + threadId;
			int symbolTwoCellId = symbolTwoId * symbolSize + threadId;

			resultSymbol[threadId] = weightOne * symbolVectors[symbolOneCellId] + weightTwo * symbolVectors[symbolTwoCellId];
		}
	
	}

	__global__ void SumSymbolsKernel(
		float *symbolOne,
		float *symbolTwo,
		float *result,
		int symbolSize
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < symbolSize)
		{
			result[threadId] = symbolOne[threadId] + symbolTwo[threadId];
		}
	}

	__global__ void SumBasicSymbolsKernel(
		float *symbolVectors,
		int symbolOneId,
		int symbolTwoId,
		float *result,
		int symbolSize
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < symbolSize)
		{
			result[threadId] = symbolVectors[symbolOneId * symbolSize + threadId] + symbolVectors[symbolTwoId * symbolSize + threadId];
		}
	}
	
	__global__ void ComputeDistanceKernel(
		float *symbolVectors,
		float *inputVector,
		float *distance,
		int symbolSize,
		int symbols
		)
	{
		int symbolId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(symbolId < symbols)
		{
			float sum = 0.00f;
			for(int i = 0; i < symbolSize; i++)
			{
				sum += symbolVectors[symbolId * symbolSize + i] * inputVector[i];
			}
			distance[symbolId] = sum;
		}
	}
}
#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

#define _SIZE_T_DEFINED 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __cplusplus 
#define __cplusplus 
#endif

extern "C"  
{
	__constant__ int D_BINS;
	__constant__ float D_MAX_VALUE;
	__constant__ float D_MIN_VALUE;
	__constant__ float D_BIN_VALUE_WIDTH;
	__constant__ int D_MEMORY_BLOCK_SIZE;

	//kernel code
	__global__ void ComputeHistogramKernel(
		
		float *globalMemData,
		int *globalHist

		)
	{
		//the kernel should be only 1D
		int globalThreadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
		int localThreadId = threadIdx.x;
		extern __shared__ int partialHist[];

		if(localThreadId < D_BINS)
		{
			//set the partial histogram in shared memory to zero
			partialHist[localThreadId] = 0;
		}
		
		__syncthreads();

		//if the global thread id is within bounds of the data array size
		if(globalThreadId < D_MEMORY_BLOCK_SIZE)
		{
			//copy the global data to local memory
			float myLocalDataValue = globalMemData[globalThreadId];
			int binIdToWrite = 0 + (D_BINS - 1) * (myLocalDataValue > D_MAX_VALUE);

			//if the local value is within limits
			if(myLocalDataValue >= D_MIN_VALUE && myLocalDataValue <= D_MAX_VALUE)
			{
				float biasedValue = myLocalDataValue - D_MIN_VALUE;

				binIdToWrite = (int)floor((double)(biasedValue/D_BIN_VALUE_WIDTH)) + 1;
				if(myLocalDataValue == D_MAX_VALUE)
				{
					binIdToWrite = D_BINS - 2;
				}
				
			}
			//write to local histogram
			atomicAdd( &(partialHist[binIdToWrite]), 1);

			__syncthreads();

			if(localThreadId < D_BINS)
			{
				//copy values to global histogam
				atomicAdd( &(globalHist[localThreadId]), partialHist[localThreadId]);
			}
		}
	}
}
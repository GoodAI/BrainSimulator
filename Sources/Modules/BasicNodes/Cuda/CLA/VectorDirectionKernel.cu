#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math_constants.h>


extern "C"  
{

	__global__ void SetImportantBitsKernel(
		float *input,
		float *output,
		float inputMin,
		float inputMax,
		float activityPercentage,
		int count
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < count)
		{
			if(input[threadId] > 0.00f)
			{
				output[threadId] = (float)(1.00f - input[threadId] / inputMax) < activityPercentage;
			}

			if(input[threadId] < 0.00f)
			{
				output[2 * threadId] = (float)(1.00f - input[threadId] / inputMin) < activityPercentage;
			}
		}
	}


	__global__ void DecodeInputKernel(
		float *input,
		float *output,
		int outputSize,
		int inputSize
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < inputSize)
		{
			// positive bits
			if(threadId < outputSize)
			{
				if(input[threadId] == 1.00f)
				{
					output[threadId] = 1.00f;
				}
				
			}
			// negative bits
			else
			{
				if(input[threadId] == 1.00f)
				{
					output[threadId - outputSize] = -1.00f;
				}
			}
		}
	}

	


	


}
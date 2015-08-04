#define _SIZE_T_DEFINED
#ifndef __cplusplus 
#define __cplusplus 
#endif

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <math_functions.h>
#include <float.h>
#include <cuComplex.h>

#define AND 0
#define OR 1
#define XOR 2

#define MAX_OPERANDS 20

extern "C"  
{
	//kernel code
	__global__ void CombineVectorsKernel(float** inputs, int inputsCount, float* output, int method, int count)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		__shared__ float* sm_inputs[MAX_OPERANDS];
		

		if (threadIdx.x < inputsCount) 
		{
			sm_inputs[threadIdx.x] = inputs[threadIdx.x];
		}
		__syncthreads();
		
		if(threadId < count)
		{			
			if (method == OR) 
			{
				float* input = sm_inputs[0];
				output[threadId] = input[threadId];

				for (int i = 1; i < inputsCount; i++) 
				{
					input = sm_inputs[i];
					output[threadId] -= input[threadId];
				}	
			}
			else {

				output[threadId] = method == XOR ? 1 : 0;

				for (int i = 0; i < inputsCount; i++) 
				{
					float* input = sm_inputs[i];
					if (method == AND) 
					{
						output[threadId] += input[threadId];
					}
					else if (method == XOR) 
					{
						output[threadId] *= input[threadId];
					}
				}			
			}
		}
	}

	//kernel code
	__global__ void CombineTwoVectorsKernel(float* input1, float* input2, float* output, int method, int count)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
		
		if(threadId < count)
		{			
			if (method == OR) 
			{				
				output[threadId] = input1[threadId] - input2[threadId];
			}
			else if (method == AND) 
			{
				output[threadId] = input1[threadId] + input2[threadId];
			}
			else if (method == XOR) 
			{
				output[threadId] = input1[threadId] * input2[threadId];
			}				
		}
	}

	__global__ void CombineTwoVectorsKernelVarSize(float* input1, float* input2, float* output, int method, int count1, int count2)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		int threadId2 = threadId < count2 ? threadId : count2 - 1;
		
		if(threadId < count1)
		{			
			if (method == OR) 
			{				
				output[threadId] = input1[threadId] - input2[threadId2];
			}
			else if (method == AND) 
			{
				output[threadId] = input1[threadId] + input2[threadId2];
			}
			else if (method == XOR) 
			{
				output[threadId] = input1[threadId] * input2[threadId2];
			}				
		}
	}

	__global__ void LengthFromElements(float* element1, float* element2, float* output, int count)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
		
		if(threadId < count)
		{
			output[threadId] = sqrtf(element1[threadId] * element1[threadId] + element2[threadId] * element2[threadId]);
		}
	}

	__global__ void MulComplexElementWise(cuFloatComplex* input1, cuFloatComplex* input2, cuFloatComplex* output, int count)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
		
		if(threadId < count)
		{			
			cuFloatComplex i1 = input1[threadId];
			cuFloatComplex i2 = input2[threadId];

			output[threadId] = cuCmulf(i1, i2);								
		}
	}

	__global__ void InvolveVector(float* input, float* output, int inputSize)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
		
		if(threadId < inputSize - 1)
		{									
			output[0] = input[0];
			output[threadId + 1] = input[inputSize - threadId - 1];
		}
	}

	__global__ void Interpolate(float* input1, float* input2, float* output, float weight, int inputSize)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
		
		if(threadId < inputSize)
		{					
			if (weight <= 0)
			{
				output[threadId] = input1[threadId];
			}
			else if (weight >= 1)
			{
				output[threadId] = input2[threadId];
			}
			else
			{
				output[threadId] = (1 - weight) * input1[threadId] + weight * input2[threadId];
			}
		}
	}
}
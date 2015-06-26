//Includes for IntelliSense 
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif


#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <cuComplex.h>

extern "C"  
{	
	//kernel code
	__global__ void InvertValuesKernel(float *input, float* outputs, int size)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		if(id < size) 
		{
			outputs[id] = 1.00f - input[id];
		}		
	}

	__global__ void InvertLengthComplexKernel(cuFloatComplex* input, cuFloatComplex* outputs, int size)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		if (id >= size)
			return;


		cuFloatComplex val = input[id];
		float length = cuCabsf(val);

		if (length < 0.00001f)
			length = 0;
		else
			length = 1 / length;

		length *= length;

		val.x *= length;
		val.y *= length;

		outputs[id] = val;
	}

	__global__ void InvertPermutationKernel(float* input, float* output, int size)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		if (id >= size)
			return;


		int temp = __float2int_rn(input[id]);

		if (input == output)
			__syncthreads();

		output[temp] = id;
	}
}

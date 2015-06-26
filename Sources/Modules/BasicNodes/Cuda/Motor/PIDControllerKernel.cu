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
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>


extern "C"  
{
    __constant__ float D_PROPORTIONAL_GAIN;
    __constant__ float D_INTEGRAL_GAIN;
    __constant__ float D_DERIVATIVE_GAIN;
	
	__constant__ float D_INTEGRAL_DECAY;

    __constant__ float D_OFFSET;
    __constant__ float D_MIN_OUTPUT;
    __constant__ float D_MAX_OUTPUT;
	__constant__ int D_COUNT;


	//kernel code
	__global__ void PIDControllerKernel(float* input, float* goal, float* output, float* previousError, float* integral)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (id < D_COUNT)
		{
			float error = input[id] - goal[id];
			integral[id] = D_INTEGRAL_DECAY * integral[id] + error;
			float derivative = error - previousError[id];

			previousError[id] = error;


			float out = D_OFFSET + D_PROPORTIONAL_GAIN * error + D_INTEGRAL_GAIN * integral[id] + D_DERIVATIVE_GAIN * derivative;
			if (out > D_MAX_OUTPUT)
				out = D_MAX_OUTPUT;
			if (out < D_MIN_OUTPUT)
				out = D_MIN_OUTPUT;

			output[id] = out;
		}
	}
}
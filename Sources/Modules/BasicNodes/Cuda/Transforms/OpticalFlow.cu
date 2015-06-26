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
#include <math.h>

extern "C" 
{
	__global__ void PrepareDerivativesKernel(float* input, float* lastInput, float* derivatives, int inputWidth, int inputHeight)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
					+ blockDim.x * blockIdx.x
					+ threadIdx.x;
		int size =  inputWidth * inputHeight;

		if (id < size) 
		{	
			float mul = 100000;
			//I_x, I_y
			float I_x = mul * derivatives[id];
			float I_y = mul * derivatives[size + id];

			//I_t
			float input_dt = mul * (input[id] - lastInput[id]);
			lastInput[id] = input[id];
			
			// I_x * I_y
			derivatives[2 * size + id] = I_x * I_y;
			// I_x * I_t
			derivatives[3 * size + id] = I_x * input_dt;
			// I_x * I_t
			derivatives[4 * size + id] = I_y * input_dt;
			// I_x ^ 2
			derivatives[id] = I_x * I_x;
			// I_y ^ 2
			derivatives[size + id] = I_y * I_y;
		}
	}

	__global__ void EvaluateVelocityKernel(float* derivatives, float* velocities, int inputWidth, int inputHeight)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
					+ blockDim.x * blockIdx.x
					+ threadIdx.x;
		int size =  inputWidth * inputHeight;

		if (id < size) 
		{	
			float I_x2 = derivatives[id];
			float I_y2 = derivatives[size + id];
			float I_xy = derivatives[2 * size + id];
			float I_xt = derivatives[3 * size + id];
			float I_yt = derivatives[4 * size + id];			

			float scalar = I_x2 * I_y2 - I_xy * I_xy;

			if (fabsf(scalar) > 0) {

				float vx = (I_xy * I_yt - I_y2 * I_xt) / scalar;
				float vy = (I_xy * I_xt - I_x2 * I_yt) / scalar;

				//float l = sqrtf(vx * vx + vy * vy);

				velocities[id] = fmaxf(fminf(vx, 1.0f), -1.0f);
				velocities[size + id] = fmaxf(fminf(vy, 1), -1);
			}
			else {
				//velocities[id] = 0;
				//velocities[size + id] = 0;
			}
		}
	}

	__global__ void FinalizeVelocityKernel(float* velocities, float* globalFlow, int inputWidth, int inputHeight)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
					+ blockDim.x * blockIdx.x
					+ threadIdx.x;
		int size =  inputWidth * inputHeight;

		if (id < size) 
		{			
			float globalFlowL = sqrtf(globalFlow[0] * globalFlow[0] + globalFlow[1] * globalFlow[1]);
			float velocityL = sqrtf(velocities[id] * velocities[id]  + velocities[size + id] * velocities[size + id]);

			if (globalFlowL > 0 && velocityL > 0) {				

				float dot = (globalFlow[0] * velocities[id] + globalFlow[1] * velocities[size + id]) / (globalFlowL * velocityL);

				if (dot > 0.7) {
					velocities[id] = 0;
					velocities[size + id] = 0;
				}
			}
		}
	}
}
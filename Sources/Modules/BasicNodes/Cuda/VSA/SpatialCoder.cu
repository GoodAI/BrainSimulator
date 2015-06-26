#define _SIZE_T_DEFINED 

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>

extern "C" 
{	
	__global__  void EncodeValues(float* values, int numOfValues, float* output, int symbolSize, int squaredMode,
		float* dirX, float* dirY,  float* negDirX, float* negDirY, float* originX, float* originY) 
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;	

		__shared__ float s_values[2];

		if (threadIdx.x < 2) 
		{
			//clamp to (-1, 1) if square mode is used

			if (squaredMode == 1) 
			{
				s_values[threadIdx.x] = fmaxf(fminf(values[threadIdx.x], 1), -1);
			}
			else 
			{
				s_values[threadIdx.x] = values[threadIdx.x];
			}
		}

		__syncthreads();
		
		if (threadId < symbolSize) 
		{				
			//origin part of symbol

			//squared: o * (1 - t)^2			
			if (squaredMode == 1) 
			{				
				output[threadId] = (1 - fabs(s_values[0])) * (1 - fabs(s_values[0])) * originX[threadId];
			}
			else 
			//linear: o * (1 - t)
			{
				output[threadId] = (1 - fabs(s_values[0])) * originX[threadId];
			}

			//direction part of symbol
			float* dir = (s_values[0] > 0) ? dirX : negDirX;			

			//squared: dir * (-t^2 + 2*t)
			if (squaredMode == 1) 
			{
				output[threadId] += (-s_values[0] * s_values[0] + 2 * fabs(s_values[0])) * dir[threadId];
			}
			//linear: dir * t
			else 
			{
				output[threadId] += fabs(s_values[0]) * dir[threadId];
			}
			
			//has Y axis?
			if (numOfValues > 1) 
			{
				//squared: o * (1 - t)^2			
				if (squaredMode == 1) 
				{				
					output[threadId] += (1 - fabs(s_values[1])) * (1 - fabs(s_values[1])) * originY[threadId];
				}
				else 
				//linear: o * (1 - t)
				{
					output[threadId] += (1 - fabs(s_values[1])) * originY[threadId];
				}			

				//direction part of symbol
				dir = (s_values[1] > 0) ? dirY : negDirY;

				//squared: dir * (-t^2 + 2*t)
				if (squaredMode == 1) 
				{
					output[threadId] += (-s_values[1] * s_values[1] + 2 * fabs(s_values[1])) * dir[threadId];
				}
				//linear: dir * t
				else 
				{
					output[threadId] += fabs(s_values[1]) * dir[threadId];
				}			
			}
		}
	}

	__global__  void DecodeValues(float* superposition, int symbolSize, float* output, float* reliability, int numOfValues, int squaredMode,
		float* dirX, float* dirY, float* negDirX, float* negDirY, float* originX, float* originY) 
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;		
	
		if (threadId < numOfValues) 
		{
			output[threadId] = 0;
			reliability[threadId] = 0;

			float* dir = threadId == 0 ? dirX : dirY;
			float* negDir = threadId == 0 ? negDirX : negDirY;
			float* origin = threadId == 0 ? originX : originY;			

			for (int i = 0; i < symbolSize; i++) 
			{
				output[threadId] += superposition[i] * dir[i] - superposition[i] * negDir[i];
				reliability[threadId] += superposition[i] * origin[i];
			}		

			reliability[threadId] += fabs(output[threadId]);
			output[threadId] /= reliability[threadId];
		}
	}
}
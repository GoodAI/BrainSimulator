#define _SIZE_T_DEFINED 

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>

extern "C"
{
	__device__ void EncodeValuesInternal(float value, float& origin, float& dir, float& output, int squaredMode)
	{
		if (squaredMode == 1)
		{
			// origin part:      o * (1 - t)^2			
			output = (1 - fabs(value)) * (1 - fabs(value)) * origin;
			// direction part:   dir * (-t^2 + 2*t)
			output += (-value * value + 2 * fabs(value)) * dir;
		}
		else
		{
			// origin part:      o * (1 - t)
			output = (1 - fabs(value)) * origin;
			// direction part:   dir * t
			output += fabs(value) * dir;
		}
	}

	__global__  void EncodeValues(float* values, int numOfValues, float* output, int symbolSize, int squaredMode,
		float* dirX, float* dirY, float* negDirX, float* negDirY, float* originX, float* originY)
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


		if (threadId >= symbolSize)
			return;


		// X dim
		float* dir = (s_values[0] > 0) ? dirX : negDirX;
		EncodeValuesInternal(s_values[0], originX[threadId], dir[threadId], output[threadId], squaredMode);

		// Y dim
		if (numOfValues > 1)
		{
			dir = (s_values[1] > 0) ? dirY : negDirY;
			EncodeValuesInternal(s_values[1], originY[threadId], dir[threadId], output[threadId], squaredMode);
		}
	}


	__global__  void DecodeValues(float* superposition, int symbolSize, float* output, float* reliability, int numOfValues, int squaredMode,
		float* dirX, float* dirY, float* negDirX, float* negDirY, float* originX, float* originY)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (threadId >= numOfValues)
			return;


		output[threadId] = 0;
		reliability[threadId] = 0;

		float* dir = threadId == 0 ? dirX : dirY;
		float* negDir = threadId == 0 ? negDirX : negDirY;
		float* origin = threadId == 0 ? originX : originY;

		for (int i = 0; i < symbolSize; i++)
		{
			// output  = s.d - s.n = s.dir
			// one of the values s.d or s.n will be (very close to) zero
			output[threadId] += superposition[i] * dir[i] - superposition[i] * negDir[i];
			// rel	   = s.o
			reliability[threadId] += superposition[i] * origin[i];
		}

		// rel     = s.o + s.dir
		reliability[threadId] += fabs(output[threadId]);
		// output  = s.dir / (s.o + s.dir)
		output[threadId] /= reliability[threadId];

		// Since s = dir*t + o*(1-t) + noise, we get
		// s.dir   = dir.dir*t + o.dir*(1-t) + dir.noise = t + 0 + dir.noise
		// s.o     = o.dir*t   + o.o*(1-t)   + o.noise   = 0 + (1-t) + o.noise
		// output  = t + dir.noise / (1 + dir.noise + o.noise)
		// Note that dir.noise and o.noise should be very close to zero.
		// This should make the decoding more precise when noise has similar dot product to dir and o.
	}
}

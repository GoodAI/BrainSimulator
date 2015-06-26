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

extern "C"  
{	
	//kernel code
	

	// Tell if the coordinates are in or out bounds
	__device__ bool isInBounds(int x, int y, int nbColumns, int nbLines)
	{
		return x >= 0 && y >= 0 && x < nbColumns && y < nbLines;
	}

	// Get the memory index of the value at coordinate (x, y)
	__device__ int getIndex(int x, int y, int nbColumns)
	{
		return x + y * nbColumns;
	}


	__global__ void EdgeDetectionKernel(float* input, float* output, int size, int nbColumns)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x
				+ blockDim.x * blockIdx.x
				+ threadIdx.x;

		// Out of bound, we quit
		if(id >= size)
			return;

		// Compute interesting values
		int nbLines = size / nbColumns;
		int dotX = id % nbColumns;
		int dotY = id / nbColumns;
		
		int nbValues = 0;
		float sum = 0;
		

		for (int dx = -1; dx <= 1; dx++)
			for (int dy = -1; dy <= 1; dy++)
				if (dx == 0 && dy == 0)
					continue;
				else
				{
					int x = dotX + dx;
					int y = dotY + dy;
					if (isInBounds(x, y, nbColumns, nbLines))
					{
						nbValues++;
						sum += - input[getIndex(x, y, nbColumns)];
					}
					__syncthreads();
				}

		output[id] = (sum + (float)nbValues * input[getIndex(dotX, dotY, nbColumns)]) / nbValues;
	}		
}
#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

extern "C"  
{
	__global__ void DrawBricksKernel(float *input, int inputWidth, int inputHeight, int* bricks, int bricksWidth, int bricksHeight, 
		float *sprite, float2 position, int2 spriteSize)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
				+ blockDim.x * blockIdx.x
				+ threadIdx.x;

		int inputSize = inputWidth * inputHeight;
		int size = spriteSize.x * spriteSize.y;
		int bricksSize = bricksWidth * bricksHeight;
		
		int px = id % spriteSize.x;
		int py = id / spriteSize.x;

		for (int i = 0; i < bricksSize; i++) 
		{
			if (bricks[i] > 0) 
			{
				int by = i / bricksWidth;
				int bx = i % bricksWidth;

				int inputOffset = ((int)position.y + by * spriteSize.y + py) * inputWidth + position.x + bx * spriteSize.x  + px;

				if (id < size && inputOffset >= 0 && inputOffset < inputSize) 
				{
					input[inputOffset] = sprite[id];
				}
			}
		}
	}
}
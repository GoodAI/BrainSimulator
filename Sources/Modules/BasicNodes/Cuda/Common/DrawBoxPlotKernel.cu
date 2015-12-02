#include <cuda.h>
#include <device_launch_parameters.h>
#include "float.h"

extern "C"
{
	__global__ void DrawBoxPlotKernel(int* box, // all vals
		int boxIdx, // actual value
		int ax, // top-left corner x  
		int ay, // top-left corner y
		int textureWidth,
		int textureHeight,
		int boxWidth,
		int boxHeight,
		float* pixels // output
		)
	{
		int x = blockIdx.x;
		int y = threadIdx.x;
		int pixidx = (x + ax) + (textureWidth * textureHeight) - (textureWidth * (y + ay));

		for (size_t i = 0; i < 5; i++)
		{
			if (y == box[boxIdx + i])
			{
				pixels[pixidx] = 0.0f; return;
			}
		}

		if ((x == 0 || x == boxWidth - 1) 
			&& box[boxIdx + 1] < y && y < box[boxIdx + 3])
		{
			pixels[pixidx] = 0.0f; return;
		}
		if (x == boxWidth / 2 
			&& ((box[boxIdx + 0] < y && y <= box[boxIdx + 1]) || (box[boxIdx + 3] < y && y <= box[boxIdx + 4])))
		{
			pixels[pixidx] = 0.0f; return;
		}
		float *faddr = &pixels[pixidx];
		*(unsigned int*)faddr = 255 << 16 | 255 << 8 | 255;
	}

	__global__ void DrawWhiteKernel(
		int textureWidth,
		int textureHeight,
		float* pixels // output
		)
	{
		int pixidx = threadIdx.x + blockIdx.x * blockDim.x;

		float *faddr = &pixels[pixidx];
		*(unsigned int*)faddr = 255 << 16 | 255 << 8 | 255;
	}
}
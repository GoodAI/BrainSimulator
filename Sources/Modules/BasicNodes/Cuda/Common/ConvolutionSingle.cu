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
	__constant__ float D_KERNEL[9];

	__device__ float mulWithKernel(int x, int y, int kx, int ky, float* input, int width, int height) 
	{		
		int px = min(max(x, 0), width - 1);
		int py = min(max(y, 0), height - 1);

		return D_KERNEL[3 * (ky + 1) + kx + 1] * input[py * width + px];

		/* CROP
		if (x >= 0 && y >= 0 && x < width && y < height) 
		{
			return D_KERNEL[3 * (ky + 1) + kx + 1] * input[y * width + x];
		}
		else 
		{
			return 0;
		}
		*/
	}
	
	__global__ void Convolution3x3Single(float* input, float* output, int width, int height)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		int size = width * height;

		if(threadId < size)
		{
			float result = 0;

			int x = threadId % width;
			int y = threadId / width;

			result += mulWithKernel(x - 1, y - 1, -1, -1, input, width, height); 
			result += mulWithKernel(x - 1, y    , -1,  0, input, width, height);
			result += mulWithKernel(x - 1, y + 1, -1,  1, input, width, height);

			result += mulWithKernel(x, y - 1, 0, -1, input, width, height);
			result += mulWithKernel(x, y    , 0,  0, input, width, height);
			result += mulWithKernel(x, y + 1, 0,  1, input, width, height);

			result += mulWithKernel(x + 1, y - 1, 1, -1, input, width, height); 
			result += mulWithKernel(x + 1, y    , 1,  0, input, width, height);
			result += mulWithKernel(x + 1, y + 1, 1,  1, input, width, height);

			output[y * width + x] = result;
		}
	}
}
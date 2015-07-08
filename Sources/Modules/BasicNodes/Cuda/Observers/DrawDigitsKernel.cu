#define _SIZE_T_DEFINED 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __cplusplus 
#define __cplusplus 
#endif

#include <cuda.h> 
#include <math_constants.h> 
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_types.h> 
#include <vector_functions.h> 


extern "C"  
{	
	__constant__ unsigned int D_BG_COLOR;
	__constant__ unsigned int D_FG_COLOR;
	__constant__ int D_IMAGE_WIDTH;
	__constant__ int D_IMAGE_HEIGHT;
	__constant__ int D_DIGIT_WIDTH;
	__constant__ int D_DIGIT_SIZE;
	__constant__ int D_DIGIT_INDEXES[100];
	__constant__ int D_DIGIT_INDEXES_LEN;
	__constant__ int D_DIGITMAP_NBCHARS;

	//kernel code
	__global__ void DrawDigitsKernel(unsigned int* canvas, float* digitBuffer, int offsetX, int offsetY)
	{		
		int id = blockDim.x*blockIdx.y*gridDim.x	
				+ blockDim.x*blockIdx.x				
				+ threadIdx.x;

		if (id >= D_DIGIT_SIZE * D_DIGIT_INDEXES_LEN)
			return;

		int digitIndex = id / D_DIGIT_SIZE;


		int globalX = offsetX + digitIndex * D_DIGIT_WIDTH + id % D_DIGIT_WIDTH;
		int globalY = offsetY + (id % D_DIGIT_SIZE) / D_DIGIT_WIDTH;
		
		if (globalX < 0 || globalY < 0 || globalX >= D_IMAGE_WIDTH || globalY >= D_IMAGE_HEIGHT)
			return;
		

		float bgRed = (D_BG_COLOR & 0x00FF0000) >> 16;
		float bgGreen = (D_BG_COLOR & 0x0000FF00) >> 8;
		float bgBlue = (D_BG_COLOR & 0x000000FF);

		float fgRed = (D_FG_COLOR & 0x00FF0000) >> 16;
		float fgGreen = (D_FG_COLOR & 0x0000FF00) >> 8;
		float fgBlue = (D_FG_COLOR & 0x000000FF);
		
		int localX = D_DIGIT_INDEXES[digitIndex] * D_DIGIT_WIDTH + id % D_DIGIT_WIDTH;
		int localY = (id % D_DIGIT_SIZE) / D_DIGIT_WIDTH;
		float factor = digitBuffer[localX + localY * D_DIGIT_WIDTH * D_DIGITMAP_NBCHARS];

		unsigned int red = (unsigned int)(factor * fgRed + (1 - factor) * bgRed);
		unsigned int green = (unsigned int)(factor * fgGreen + (1 - factor) * bgGreen);
		unsigned int blue = (unsigned int)(factor * fgBlue + (1 - factor) * bgBlue);
		
		unsigned int value = 0xFF000000 + (red << 16) + (green << 8) + blue;

		if (factor > 0 || D_BG_COLOR & 0xFF000000 != 0) 
		{
			canvas[globalX + globalY * D_IMAGE_WIDTH] = value;
		}
	}
}
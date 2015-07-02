#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

/*
Inspired by the implementation of CustomPong.cu

@author jv
*/
extern "C"  
{

	/*
	Draws entire map
	inputWidth & inputHeight: map dimensions in pixels
	talesWidth & height: no of tales
	*/
	__global__ void DrawTalesKernel(float *input, int inputWidth, int inputHeight,
		int* tiles, int tilesWidth, int tilesHeight, 
		float *sprite, float *obstacleSprite, int2 spriteSize)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		int inputSize = inputWidth * inputHeight;
		int size = spriteSize.x * spriteSize.y;
		int tilesSize = tilesWidth * tilesHeight;

		int spriteId = id % size;

		int px = spriteId % spriteSize.x;
		int py = spriteId / spriteSize.x;

		int tileId = id / size;

		// position of my tale
		int by = tileId / tilesWidth;
		int bx = tileId % tilesWidth;

		// original one, not upside down
		//int inputOffset = ((int)by * spriteSize.y + py) * inputWidth + bx * spriteSize.x  + px;
		int inputOffset = ((tilesHeight-1-by) * spriteSize.y + py) * inputWidth 
			+ bx * spriteSize.x  + px;

		if (id < inputSize && inputOffset >= 0 && inputOffset < inputSize) 
		{
			// obstacles are marked as 1
			if(tiles[tileId] == 1)
			{
				input[inputOffset] = obstacleSprite[spriteId];
			}
			// everything else will be drawn as free and you can place anything over it
			else
			{
				input[inputOffset] = sprite[spriteId];
			}
		}
		
	}
	
	/*
	position: in tale coordinates
	resolution: tale size in pixels
	inputWidth: width of the visible area
	*/
	__global__ void DrawObjectKernel(float *input, int resolution, int inputWidth, int inputHeight, 
		float *sprite, int2 position, int2 spriteSize)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		int inputSize = inputWidth * inputHeight;
		int size = spriteSize.x * spriteSize.y;

		int px = id % spriteSize.x;
		int py = id / spriteSize.x;

		// where to draw a pixel in the visual array
		//int inputOffset = (position.y*resolution+ py) * inputWidth + (resolution*position.x + px);
		// upside down version
		int talesHeight = inputHeight/resolution;
		int inputOffset = ((talesHeight-1-position.y) * resolution + py) * inputWidth
			+ resolution*position.x + px;
		
		if (id < size && inputOffset >= 0 && inputOffset < inputSize && sprite[id] < 1.0f) 
		{
			input[inputOffset] = sprite[id];
		}
	}

	__global__ void DrawFreeObjectKernel(float *input, int inputWidth, int inputHeight, 
		float *sprite, int2 position, int2 spriteSize)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		int inputSize = inputWidth * inputHeight;
		int size = spriteSize.x * spriteSize.y;

		int px = id % spriteSize.x;
		int py = id / spriteSize.x;
		
		int inputOffset = (inputHeight - 1 - position.y + py) * inputWidth + position.x + px;
		
		if (id < size && inputOffset >= 0 && inputOffset < inputSize && sprite[id] < 1.0f) 
		{
			input[inputOffset] = sprite[id];
		}
	}
}

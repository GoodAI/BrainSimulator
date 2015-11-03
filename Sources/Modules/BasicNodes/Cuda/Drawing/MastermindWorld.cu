#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

/*
Inspired by the implementation of CustomPong.cu + GridWorld.cu

@author mp
*/
extern "C"
{

	/*
	Draws a background color into a 3-component image.
	inputWidth & inputHeight: map dimensions in pixels
	*/
	__global__ void DrawRgbBackgroundKernel(float *target, int inputWidth, int inputHeight,
		float r, float g, float b)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		int imagePixels = inputWidth * inputHeight; 

		if (id < imagePixels * 3) // 3 for RGB 
		{
			float color = 0.0f;
			switch (id / imagePixels)
			{
			case 0:
				color = r;
				break;
			case 1:
				color = g;
				break;
			case 2:
				color = b;
				break;
			}
			target[id] = color;
		}
	}

	/*
	Draws a texture into a 3-component target. RGBA. Checks bounds.
	*/
	__global__ void DrawRgbaTextureKernel(float *target, int targetWidth, int targetHeight, int inputX, int inputY,
		float *texture, int textureWidth, int textureHeight)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x 
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		int targetPixels = targetWidth * targetHeight;

		int texturePixels = textureWidth * textureHeight;

		int idTextureRgb = id / texturePixels;
		int idTexturePixel = (id - idTextureRgb * texturePixels); // same as (id % texturePixels), but the kernel runs 10% faster
		int idTextureY = idTexturePixel / textureWidth; 
		int idTextureX = (idTexturePixel - idTextureY * textureWidth); // same as (id % textureWidth), but the kernel runs another 10% faster


		if (idTextureRgb < 3) // 3 channels that we will write to
		{
			// the texture is in BGR format, we want RGB
			switch (idTextureRgb)
			{
			case 0: // R
				idTextureRgb = 2; // B
				break;
			case 2: // B
				idTextureRgb = 0; // R
				break;
			}
			// if the texture pixel offset by inputX, inputY, lies inside the target
			if (idTextureX + inputX < targetWidth &&
				idTextureX + inputX >= 0 &&
				idTextureY + inputY < targetHeight &&
				idTextureY + inputY >= 0)
			{
				int tIndex = targetPixels * idTextureRgb + targetWidth * (idTextureY + inputY) + (idTextureX + inputX);
				int aIndex = idTexturePixel + 3 * texturePixels; // the A component of the texture
				float a = texture[aIndex];
				target[tIndex] = target[tIndex] * (1.0f - a) + a * texture[id];
			}
		}
	}

	/*
	Optimized version of DrawRgbaTextureKernel : avoids division operations (~30% speedup)
	The width of the texture is in blockDim.x
	The height of the texture is distributed between blockDim.y and gridDim.x
	*/
	__global__ void DrawRgbaTextureKernel2DBlock(float *target, int targetWidth, int targetHeight, int inputX, int inputY,
		float *texture, int textureWidth, int textureHeight)
	{
		int id = blockDim.x * blockDim.y * (blockIdx.y * gridDim.x + blockIdx.x)
			+ blockDim.x * threadIdx.y
			+ threadIdx.x; // 2D grid of 2D blocks; block dimension x = texture width; 
		// grid dimension x + block dimension y = texture height

		int targetPixels = targetWidth * targetHeight;

		int texturePixels = textureWidth * textureHeight;

		int idTextureRgb = blockIdx.y;
		int idTexturePixel = (id - idTextureRgb * texturePixels);
		int idTextureY = blockIdx.x * blockDim.y + threadIdx.y;
		int idTextureX = threadIdx.x;


		if (idTextureRgb < 3) // 3 channels that we will write to
		{
			// the texture is in BGR format, we want RGB
			switch (idTextureRgb)
			{
			case 0: // R
				idTextureRgb = 2; // B
				break;
			case 2: // B
				idTextureRgb = 0; // R
				break;
			}
			// if the texture pixel offset by inputX, inputY, lies inside the target
			if (idTextureX + inputX < targetWidth &&
				idTextureX + inputX >= 0 &&
				idTextureY + inputY < targetHeight &&
				idTextureY + inputY >= 0)
			{
				int tIndex = targetPixels * idTextureRgb + targetWidth * (idTextureY + inputY) + (idTextureX + inputX);
				int aIndex = idTexturePixel + 3 * texturePixels; // the A component of the texture
				float a = texture[aIndex];
				target[tIndex] = target[tIndex] * (1.0f - a) + a * texture[id];
			}
		}
	}

	/*
	Draws an RGB color into the masked area. The color is drawn in each pixel that has non-0 alpha.
	*/
	__global__ void DrawMaskedColorKernel(float *target, int targetWidth, int targetHeight, int inputX, int inputY,
		float *textureMask, int textureWidth, int textureHeight, float r, float g, float b) 
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		int targetPixels = targetWidth * targetHeight;

		int texturePixels = textureWidth * textureHeight;

		int idTextureRgb = id / texturePixels;
		int idTexturePixel = (id - idTextureRgb * texturePixels); // same as (id % texturePixels), but the kernel runs 10% faster
		int idTextureY = idTexturePixel / textureWidth;
		int idTextureX = (idTexturePixel - idTextureY * textureWidth); // same as (id % textureWidth), but the kernel runs another 10% faster

		if (idTextureRgb < 3) // only RGB channels are interesting
		{
			// if the texture pixel offset by inputX, inputY, lies inside the target
			if (idTextureX + inputX < targetWidth &&
				idTextureX + inputX >= 0 &&
				idTextureY + inputY < targetHeight &&
				idTextureY + inputY >= 0)
			{
				int tIndex = targetPixels * idTextureRgb + targetWidth * (idTextureY + inputY) + (idTextureX + inputX);
				int aIndex = idTexturePixel + 3 * texturePixels; // the A component of the texture
				float a = textureMask[aIndex];

				if (a > 0) // mask allows color here
				{
					switch (idTextureRgb)
					{
					case 0:
						target[tIndex] = r;
						break;
					case 1:
						target[tIndex] = g;
						break;
					case 2:
					default:
						target[tIndex] = b;
						break;
					}
				}
			}
		}
	}

	/*
	Optimized version of DrawMaskedColorKernel : avoids division operations (~30% speedup)
	The width of the texture is in blockDim.x
	The height of the texture is distributed between blockDim.y and gridDim.x
	*/
	__global__ void DrawMaskedColorKernel2DBlock(float *target, int targetWidth, int targetHeight, int inputX, int inputY,
		float *textureMask, int textureWidth, int textureHeight, float r, float g, float b)
	{
		int id = blockDim.x * blockDim.y * (blockIdx.y * gridDim.x + blockIdx.x)
			+ blockDim.x * threadIdx.y
			+ threadIdx.x; // 2D grid of 2D blocks; block dimension x = texture width; 
		// grid dimension x + block dimension y = texture height

		int targetPixels = targetWidth * targetHeight;

		int texturePixels = textureWidth * textureHeight;

		int idTextureRgb = blockIdx.y;
		int idTexturePixel = (id - idTextureRgb * texturePixels);
		int idTextureY = blockIdx.x * blockDim.y + threadIdx.y;
		int idTextureX = threadIdx.x;


		if (idTextureRgb < 3) // only RGB channels are interesting
		{
			// if the texture pixel offset by inputX, inputY, lies inside the target
			if (idTextureX + inputX < targetWidth &&
				idTextureX + inputX >= 0 &&
				idTextureY + inputY < targetHeight &&
				idTextureY + inputY >= 0)
			{
				int tIndex = targetPixels * idTextureRgb + targetWidth * (idTextureY + inputY) + (idTextureX + inputX);
				int aIndex = idTexturePixel + 3 * texturePixels; // the A component of the texture
				float a = textureMask[aIndex];

				if (a > 0) // mask allows color here
				{
					switch (idTextureRgb)
					{
					case 0:
						target[tIndex] = r;
						break;
					case 1:
						target[tIndex] = g;
						break;
					case 2:
					default:
						target[tIndex] = b;
						break;
					}
				}
			}
		}
	}

}

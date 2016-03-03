#include <cuda.h>
#include <device_launch_parameters.h>

/*
Inspired by the implementation of CustomPong.cu + GridWorld.cu

@author mp
*/
extern "C"
{

	/*
	Draws a background color into a 3-component image.
	inputWidth & inputHeight: map dimensions in pixels
	gridDim.y = 3, one for each color component
	*/
	__global__ void DrawRgbBackgroundKernel(float *target, int inputWidth, int inputHeight,
		float r, float g, float b)
	{
		int column = threadIdx.x + blockDim.x * blockIdx.z;
		if (column >= inputWidth)
			return;

		int id = inputWidth * ( blockIdx.y * gridDim.x + blockIdx.x) // blockIdx.x == row, blockIdx.y == color channel 
			+ column;

		int imagePixels = inputWidth * inputHeight; 

		if (id < 3*imagePixels) // 3 for RGB 
		{
			float color = 0.0f;
			switch (blockIdx.y)
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
	Adds noise into a 3-component image.
	inputWidth & inputHeight: map dimensions in pixels
	*/
	__global__ void AddRgbNoiseKernel(float *target, int inputWidth, int inputHeight, float *randoms)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		int imagePixels = inputWidth * inputHeight;

		if (id < imagePixels)
		{
			unsigned int tg = *((unsigned int*)(&target[id]));

			int blue = (tg >> 0) & (0xFF);
			blue += (int)(randoms[id]);
			blue = blue < 255 ? blue : 255;
			blue = blue > 0 ? blue : 0;

			int green = ((tg >> 8) & (0xFF));
			green += (int)(randoms[id + imagePixels]);
			green = green < 255 ? green : 255;
			green = green > 0 ? green : 0;

			int red = ((tg >> 16) & (0xFF));
			red += (int)(randoms[id + imagePixels * 2]);
			red = red < 255 ? red : 255;
			red = red > 0 ? red : 0;

			// alpha is the last channel (<< 24)
			unsigned int tmp = (*((unsigned int *)(&blue)) << 0)
						     | (*((unsigned int *)(&green)) << 8)
							 | (*((unsigned int *)(&red)) << 16);

			target[id] = *((float *)(&tmp));
		}
	}

	/* Fill specified rectangle with color */
	__global__ void DrawRgbaColorKernel(float *target, int targetWidth, int targetHeight, int inputX, int inputY,
		int areaWidth, int areaHeight, float r, float g, float b)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		int targetPixels = targetWidth * targetHeight;

		int texturePixels = areaWidth * areaHeight;

		int idTextureRgb = id / texturePixels;
		int idTexturePixel = (id - idTextureRgb * texturePixels); // same as (id % texturePixels), but the kernel runs 10% faster
		int idTextureY = idTexturePixel / areaWidth;
		int idTextureX = (idTexturePixel - idTextureY * areaWidth); // same as (id % textureWidth), but the kernel runs another 10% faster


		if (idTextureRgb < 3) // 3 channels that we will write to
		{
			// if the texture pixel offset by inputX, inputY, lies inside the target
			if (idTextureX + inputX < targetWidth &&
				idTextureX + inputX >= 0 &&
				idTextureY + inputY < targetHeight &&
				idTextureY + inputY >= 0)
			{
				float color = 0.0f;
				switch (idTextureRgb)
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
				int tIndex = targetPixels * idTextureRgb + targetWidth * (idTextureY + inputY) + (idTextureX + inputX);
				target[tIndex] = color;
			}
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
	Draws a texture into a 3-component target. RGBA. Checks bounds. Stretches the texture.
	*/
	__global__ void DrawRgbaTextureKernelNearestNeighbor(float *target, int targetWidth, int targetHeight, int inputX, int inputY,
		float *texture, int textureWidth, int textureHeight, int objectWidth, int objectHeight)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		int targetPixels = targetWidth * targetHeight;

		int texturePixels = textureWidth * textureHeight;

		int objectPixels = objectWidth * objectHeight;

		int idObjectRgb = id / objectPixels;
		int idObjectPixel = (id - idObjectRgb * objectPixels); // same as (id % objectPixels), but the kernel runs 10% faster
		int idObjectY = idObjectPixel / objectWidth;
		int idObjectX = (idObjectPixel - idObjectY * objectWidth); // same as (id % textureWidth), but the kernel runs another 10% faster


		if (idObjectRgb < 3) // 3 channels that we will write to
		{
			int targetRgb = idObjectRgb;
			// the texture is in BGR format, we want RGB
			switch (idObjectRgb)
			{
			case 0: // R
				targetRgb = 2; // B
				break;
			case 2: // B
				targetRgb = 0; // R
				break;
			}
			// if the object pixel offset by inputX, inputY, lies inside the target
			if (idObjectX + inputX < targetWidth &&
				idObjectX + inputX >= 0 &&
				idObjectY + inputY < targetHeight &&
				idObjectY + inputY >= 0)
			{
				// nearest neighbor texture X,Y:
				int textureX = textureWidth * idObjectX / objectWidth;
				int textureY = textureHeight * idObjectY / objectHeight;
				int textureId = textureY * textureWidth + textureX;
				
				int rgbIndex = textureId + idObjectRgb * texturePixels;
				float textureValue = texture[rgbIndex];

				int tIndex = targetPixels * targetRgb + targetWidth * (idObjectY + inputY) + (idObjectX + inputX);
				int aIndex = textureId + 3 * texturePixels; // the A component of the texture
				float a = texture[aIndex];
				target[tIndex] = target[tIndex] * (1.0f - a) + a * textureValue;
			}
		}
	}

	/*
	Same as DrawRgbaTextureKernelNearestNeighbor, but texture = mask and texture's pixel values are replaced by a single color
	*/
	__global__ void DrawMaskedColorKernelNearestNeighbor(float *target, int targetWidth, int targetHeight, int inputX, int inputY,
		float *texture, int textureWidth, int textureHeight, int objectWidth, int objectHeight, 
		float r, float g, float b ) // texture = mask
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		int targetPixels = targetWidth * targetHeight;

		int texturePixels = textureWidth * textureHeight;

		int objectPixels = objectWidth * objectHeight;

		int idObjectRgb = id / objectPixels;
		int idObjectPixel = (id - idObjectRgb * objectPixels); // same as (id % objectPixels), but the kernel runs 10% faster
		int idObjectY = idObjectPixel / objectWidth;
		int idObjectX = (idObjectPixel - idObjectY * objectWidth); // same as (id % textureWidth), but the kernel runs another 10% faster


		if (idObjectRgb < 3) // 3 channels that we will write to
		{
			int targetRgb = idObjectRgb;
			// the texture is in BGR format, we want RGB
			switch (idObjectRgb)
			{
			case 0: // R
				targetRgb = 2; // B
				break;
			case 2: // B
				targetRgb = 0; // R
				break;
			}
			// if the object pixel offset by inputX, inputY, lies inside the target
			if (idObjectX + inputX < targetWidth &&
				idObjectX + inputX >= 0 &&
				idObjectY + inputY < targetHeight &&
				idObjectY + inputY >= 0)
			{
				// nearest neighbor texture X,Y:
				int textureX = textureWidth * idObjectX / objectWidth;
				int textureY = textureHeight * idObjectY / objectHeight;
				int textureId = textureY * textureWidth + textureX;

				int tIndex = targetPixels * targetRgb + targetWidth * (idObjectY + inputY) + (idObjectX + inputX);
				int aIndex = textureId + 3 * texturePixels; // the A component of the texture
				float a = texture[aIndex];

				if (a > 0) // mask allows color here
				{
					// apply this: target[tIndex] = target[tIndex] * (1.0f - a) + a * color;
					target[tIndex] = target[tIndex] * (1.0f - a);
					switch (idObjectRgb)
					{
					case 0:
						target[tIndex] += a*r;
						break;
					case 1:
						target[tIndex] += a*g;
						break;
					case 2:
					default:
						target[tIndex] += a*b;
						break;
					}
				}
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

	/*
	Convert Raw to RGB
	*/
	__global__ void ExtractRawComponentsToRgbKernel(float *target, int inputWidth, int inputHeight)
	{
		int pixelId = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		int imagePixels = inputWidth * inputHeight;

		if (pixelId >= imagePixels)
			return;

		unsigned int* uTarget = (unsigned int*)target;

		for (int i = 2; i >= 0; i--)
		{
			unsigned int component = uTarget[pixelId];
			component = component >> (8 * (2-i)); // 2-i == RGB -> BGR
			component = component & 0xFF;
			target[imagePixels * i + pixelId] = ((float)component)/255.0f;
			__syncthreads();
		}
	}

}

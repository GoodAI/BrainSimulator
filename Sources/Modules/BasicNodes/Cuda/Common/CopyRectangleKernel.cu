#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>


extern "C"  
{
	// 1 thread = 1 pixel

	//kernel code
	__global__ void CopyRectangleKernel(	float *src, int srcOffset, int srcWidth, int srcRectX, int srcRectY,
											int rectWidth, int rectHeight,
											float *dest, int destOffset, int destWidth, int destRectX, int destRectY
									)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		int size = rectWidth * rectHeight; 
		
		if (id < size) {			

			int localX = id % rectWidth;
			int localY = id / rectWidth;

			int srcPixelX = srcRectX + localX;
			int srcPixelY = srcRectY + localY;

			int destPixelX = destRectX + localX;
			int destPixelY = destRectY + localY;
			
			(dest + destOffset)[destPixelX + destPixelY * destWidth] = (src + srcOffset)[srcPixelX + srcPixelY * srcWidth];
		}
	}

	__global__ void CopyRectangleCheckBoundsKernel(	float *src, int srcOffset, int srcWidth, int srcHeight, int srcRectX, int srcRectY,
											int rectWidth, int rectHeight,
											float *dest, int destOffset, int destWidth, int destRectX, int destRectY, float defaultValue
									)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		int size = rectWidth * rectHeight; 
		
		if (id < size) {			

			int localX = id % rectWidth;
			int localY = id / rectWidth;

			int srcPixelX = srcRectX + localX;
			int srcPixelY = srcRectY + localY;

			int destPixelX = destRectX + localX;
			int destPixelY = destRectY + localY;

			if (srcPixelX >= 0 && srcPixelX < srcWidth && srcPixelY >= 0 && srcPixelY < srcHeight) 
			{
				(dest + destOffset)[destPixelX + destPixelY * destWidth] = (src + srcOffset)[srcPixelX + srcPixelY * srcWidth];
			}
			else 
			{
				(dest + destOffset)[destPixelX + destPixelY * destWidth] = defaultValue;
			}
		}
	}
}
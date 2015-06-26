
#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

extern "C"  
{
	__global__ void BilinearResampleKernel(float *input, float *output, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
				+ blockDim.x * blockIdx.x
				+ threadIdx.x;
		int size =  outputWidth * outputHeight;

		if (id < size) 
		{
			int px = id % outputWidth;
			int py = id / outputWidth;

			float xRatio = (float)(inputWidth - 1) / (outputWidth);
			float yRatio = (float)(inputHeight - 1) / (outputHeight);

			int x = (int) (xRatio * (px+.5f));
			int y = (int) (yRatio * (py+.5f));          
 
			// X and Y distance difference
			float xDist = (xRatio * (px+.5f)) - x+.5f;
			float yDist = (yRatio * (py+.5f)) - y+.5f;
 
			// Points
			float topLeft = input[y * inputWidth + x];
			float topRight = input[y * inputWidth + x + 1];
			float bottomLeft = input[(y + 1) * inputWidth + x];
			float bottomRight = input[(y + 1) * inputWidth + x + 1]; 
                
			float result = 
				topLeft * (1 - xDist) * (1 - yDist) + 
				topRight * xDist * (1 - yDist) + 
				bottomLeft * yDist * (1 - xDist) + 
				bottomRight * xDist * yDist;
 
			output[py * outputWidth + px] = result;
		}
	}




	__global__ void NNResampleKernel(float *input, float *output, int inputWidth, int inputHeight, int outputWidth, int outputHeight)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;
		int size =  outputWidth * outputHeight;

		if (id < size) 
		{
			int px = id % outputWidth;
			int py = id / outputWidth;

			float xRatio = (float)(inputWidth - 1) / (outputWidth);
			float yRatio = (float)(inputHeight - 1) / (outputHeight);

			int x = (int) (xRatio * (px+.5f));
			int y = (int) (yRatio * (py+.5f));           

			output[py * outputWidth + px] = input[y*inputWidth + x];
		}
	}




	__global__ void BilinearResampleSubImageKernel(float *input, float *output, float* subImageDefs, bool safeBounds,
		int inputWidth, int inputHeight, int outputWidth, int outputHeight)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
				+ blockDim.x * blockIdx.x
				+ threadIdx.x;
		int size =  outputWidth * outputHeight;

		if (id < size) 
		{
			float subImgCX = subImageDefs[0]; // <-1, 1>
			float subImgCY = subImageDefs[1]; // <-1, 1>
			float subImgDiameter = subImageDefs[2]; // <0,1>

			int maxDiameter = min(inputWidth - 1, inputHeight - 1);
			int diameterPix = (int)(subImgDiameter * maxDiameter);

			diameterPix = max(1, diameterPix);
			diameterPix = min(maxDiameter, diameterPix);

			int subImgX = (int)(inputWidth * (subImgCX + 1) * 0.5f) - diameterPix / 2;
			int subImgY = (int)(inputHeight * (subImgCY + 1) * 0.5f) - diameterPix / 2;

			if (safeBounds) 
			{
				subImgX = max(subImgX, 1);
				subImgY = max(subImgY, 1);

				subImgX = min(subImgX, inputWidth - diameterPix - 1);
				subImgY = min(subImgY, inputHeight - diameterPix - 1);			
			}

			int px = id % outputWidth;
			int py = id / outputWidth;
				
			float xRatio = (float)(diameterPix - 1) / (outputWidth - 1);
			float yRatio = (float)(diameterPix - 1) / (outputHeight - 1);

			int x = (int) (xRatio * px);
			int y = (int) (yRatio * py);   

			if (x + subImgX >= 0 && y + subImgY >= 0 &&
				x + subImgX < inputWidth && y + subImgY < inputHeight) 
			{
				// X and Y distance difference
				float xDist = (xRatio * px) - x;
				float yDist = (yRatio * py) - y;
 
				// Points
				float topLeft= input[(y + subImgY) * inputWidth + x + subImgX];
				float topRight = input[(y + subImgY) * inputWidth + x + subImgX + 1];
				float bottomLeft = input[(y + subImgY + 1) * inputWidth + x + subImgX];
				float bottomRight = input[(y + subImgY + 1) * inputWidth + x + subImgX + 1]; 
                
				float result = 
					topLeft * (1 - xDist) * (1 - yDist) + 
					topRight * xDist * (1 - yDist) + 
					bottomLeft * yDist * (1 - xDist) + 
					bottomRight * xDist * yDist;
 
				output[py * outputWidth + px] = result;
			}
		}
	}




    ///  Resmaple for the set of locations. It needs proper 
   	__global__ void BilinearResampleSubImageKernel_ForManyProposals(const float *input, float *output, const float* subImageDefs, bool safeBounds,
		int subImageDefsDim, int inputWidth, int inputHeight, int outputWidth, int outputHeight, int numberSubImages, int outputSize)
	{

        int px = threadIdx.x;
        int py = threadIdx.y;
        int subim_id = blockIdx.x;

        //__shared__ float cache[6*6] ;  //--- it does not make a big difference :-(

        if (blockDim.x*blockDim.y*gridDim.x != outputSize)
        {
            //--- wrong kernel sizes!!!
            return;
        }

        if ((px + py*outputWidth + subim_id*outputWidth*outputHeight)<outputSize) // double check!!
        {

		//---- copy of subimCode to resample image
			float subImgCX = subImageDefs[0 + subim_id*subImageDefsDim]; // <-1, 1>
			float subImgCY = subImageDefs[1 + subim_id*subImageDefsDim]; // <-1, 1>
			float subImgDiameter = subImageDefs[2 + subim_id*subImageDefsDim]; // <0,1>

			int maxDiameter = min(inputWidth - 1, inputHeight - 1);
			int diameterPix = (int)(subImgDiameter * maxDiameter);

			diameterPix = max(1, diameterPix);
			diameterPix = min(maxDiameter, diameterPix);

			int subImgX = (int)(inputWidth * (subImgCX + 1) * 0.5f) - diameterPix / 2;
			int subImgY = (int)(inputHeight * (subImgCY + 1) * 0.5f) - diameterPix / 2;

			if (safeBounds) 
			{
				subImgX = max(subImgX, 1);
				subImgY = max(subImgY, 1);

				subImgX = min(subImgX, inputWidth - diameterPix - 1);
				subImgY = min(subImgY, inputHeight - diameterPix - 1);			
			}

			float xRatio = (float)(diameterPix - 1) / (outputWidth - 1);
			float yRatio = (float)(diameterPix - 1) / (outputHeight - 1);

			int x = (int) (xRatio * px);
			int y = (int) (yRatio * py);   

			if (x + subImgX >= 0 && y + subImgY >= 0 &&
				x + subImgX < inputWidth && y + subImgY < inputHeight) 
			{
				//--- X and Y distance difference
				float xDist = (xRatio * px) - x;
				float yDist = (yRatio * py) - y;
 
				//--- Points
				float topLeft= input[(y + subImgY) * inputWidth + x];
				float topRight = input[(y + subImgY) * inputWidth + x + subImgX + 1];
				float bottomLeft = input[(y + subImgY + 1) * inputWidth + x + subImgX];
				float bottomRight = input[(y + subImgY + 1) * inputWidth + x + subImgX + 1 ]; 
                
				float result = 
					topLeft * (1 - xDist) * (1 - yDist) + 
					topRight * xDist * (1 - yDist) + 
					bottomLeft * yDist * (1 - xDist) + 
					bottomRight * xDist * yDist;
 
				output[py * outputWidth + px + subim_id*outputWidth*outputHeight] = result;
                //cache[py * outputWidth + px] = result;
			}
	    /*	//--- copy these results to the output matrix :D
            __syncthreads();
	    	if (px==0 && py==0){
                for (int i=0 ; i<outputWidth*outputHeight ; i++)
                {
                    output[i + subim_id*outputWidth*outputHeight] = cache[i];
                }
            }
         */
        }
	}





	__global__ void BilinearAddSubImageKernel(float *input, float *opImage, float* subImageDefs, int inputWidth, int inputHeight, int opImageWidth, int opImageHeight)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
				+ blockDim.x * blockIdx.x
				+ threadIdx.x;		

		float subImgCX = subImageDefs[0]; // <-1, 1>
		float subImgCY = subImageDefs[1]; // <-1, 1>
		float subImgDiameter = subImageDefs[2]; // <0,1>

		int maxDiameter = min(inputWidth, inputHeight);
		int diameterPix = (int)(subImgDiameter * maxDiameter);
		diameterPix = max(1, diameterPix);

		int subImgX = (int)(inputWidth * (subImgCX + 1) * 0.5f) - diameterPix / 2;
		int subImgY = (int)(inputHeight * (subImgCY + 1) * 0.5f) - diameterPix / 2;

		int px = id % diameterPix;
		int py = id / diameterPix;

		if (px + subImgX >= 0 && py + subImgY >= 0 &&
			px + subImgX < inputWidth && py + subImgY < inputHeight &&
			py < diameterPix ) 
		{						
			float xRatio = (float)(opImageWidth - 1) / (diameterPix);
			float yRatio = (float)(opImageHeight - 1) / (diameterPix);

			int x = (int) (xRatio * px);
			int y = (int) (yRatio * py);          
 
			// X and Y distance difference
			float xDist = (xRatio * px) - x;
			float yDist = (yRatio * py) - y;
 
			// Points
			float topLeft= opImage[y * opImageWidth + x];
			float topRight = opImage[y * opImageWidth + x + 1];
			float bottomLeft = opImage[(y + 1) * opImageWidth + x];
			float bottomRight = opImage[(y + 1) * opImageWidth + x + 1]; 
                
			float result = 
				topLeft * (1 - xDist) * (1 - yDist) + 
				topRight * xDist * (1 - yDist) + 
				bottomLeft * yDist * (1 - xDist) + 
				bottomRight * xDist * yDist;
				
  
			input[(py + subImgY) * inputWidth + px + subImgX] += result;
		}
	}

	__global__ void DrawSpriteKernel(float *input, int inputWidth, int inputHeight, float *sprite, float2 position, int2 spriteSize)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
				+ blockDim.x * blockIdx.x
				+ threadIdx.x;

		int inputSize = inputWidth * inputHeight;
		int size = spriteSize.x * spriteSize.y;
		
		int px = id % spriteSize.x;
		int py = id / spriteSize.x;

		int inputOffset = ((int)position.y + py) * inputWidth + position.x + px;

		if (id < size && inputOffset >= 0 && inputOffset < inputSize) 
		{
			input[inputOffset] = sprite[id];
		}
	}

	__global__ void Crop2DKernel(float *input, float *output, int inputWidth, int inputHeight, int outputWidth, int size, int leftMargin, int topMargin, float fillValue)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
				+ blockDim.x * blockIdx.x
				+ threadIdx.x;

		if (id < size) 
		{
			int inputX = id % outputWidth - leftMargin;
			int inputY = id / outputWidth - topMargin;

			if (inputX >= 0 && inputX < inputWidth && inputY >= 0 && inputY < inputHeight)
				output[id] = input[inputX + inputY * inputWidth];
			else
				output[id] = fillValue;
		}
	}

}
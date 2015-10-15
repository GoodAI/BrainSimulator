
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
		int id = blockDim.x * blockIdx.y * gridDim.x
				+ blockDim.x * blockIdx.x
				+ threadIdx.x;

        int px = id % outputWidth;  // line in the single output image
        int subim_id = id / outputWidth / outputHeight;  // which image it is
        int py = (id / outputWidth) % outputHeight;  // column in the single output image

        if (id<outputSize)
        {
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
				float topLeft= input[(y + subImgY) * inputWidth + x + subImgX];
				float topRight = input[(y + subImgY) * inputWidth + x + subImgX + 1];
				float bottomLeft = input[(y + subImgY + 1) * inputWidth + x + subImgX];
				float bottomRight = input[(y + subImgY + 1) * inputWidth + x + subImgX + 1 ]; 
                
				float result = 
					topLeft * (1 - xDist) * (1 - yDist) + 
					topRight * xDist * (1 - yDist) + 
					bottomLeft * yDist * (1 - xDist) + 
					bottomRight * xDist * yDist;
 
				output[py * outputWidth + px + subim_id*outputWidth*outputHeight] = result;
			}
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





    //------------------------------------------------------------------------------------------------------------------------
    //                          RETINA STUFF
    //------------------------------------------------------------------------------------------------------------------------

    __device__ void EstimateParForSubsample(float* subImageDefs, bool safeBounds,
		int inputWidth, int inputHeight,
        int2 & subImg, int & diameterPix)
    {
    	diameterPix = (int)( fminf( (float)inputWidth,(float)inputHeight ) * subImageDefs[2] ); // <0,1> 

		subImg.x = (int)((float)inputWidth * (subImageDefs[0] + 1) * 0.5f) ;//- diameterPix / 2;
		subImg.y = (int)((float)inputHeight * (subImageDefs[1] + 1) * 0.5f);// - diameterPix / 2;

		int maxDiameter = min(inputWidth - 1, inputHeight - 1);

        diameterPix = max(1, diameterPix);
		diameterPix = min(maxDiameter, diameterPix);

		if (safeBounds) 
		{
			subImg.x = max(subImg.x, 1);
			subImg.y = max(subImg.y, 1);
			subImg.x = min(subImg.x, inputWidth - diameterPix - 1);
			subImg.y = min(subImg.y, inputHeight - diameterPix - 1);			
		}
    }


    __global__ void RetinaTransform_HaveAtLeastOneValueThere (float * subImageDefs, 
                                                     float* input, int inputWidth, int inputHeight,
                                                     float* output,int outputDataSize,
                                                     float* retinaMask, int retinaDataSize, int retinaMaskColHint,
                                                     float* retinaDataInserted)
    {
        int id_retinaPoint = blockDim.x * blockIdx.y * gridDim.x
				    + blockDim.x * blockIdx.x
				    + threadIdx.x;

		int2 subImg;
        int diameterPix;
        bool  safeBounds = 0;


        EstimateParForSubsample( subImageDefs,  safeBounds, inputWidth,  inputHeight,  subImg, diameterPix );

        if (id_retinaPoint<outputDataSize)
        {
            output[id_retinaPoint] = 0; // default value
            float x_mask = (retinaMask[id_retinaPoint*retinaMaskColHint]*diameterPix);
            float y_mask = (retinaMask[id_retinaPoint*retinaMaskColHint+1]*diameterPix);

            int x = subImg.x + x_mask;
            int y = subImg.y + y_mask;
            if (x<inputWidth && y<inputHeight && x>=0 && y>=0)
            {
                float val = input[x+y*inputWidth];
                output[id_retinaPoint] = val;

                atomicAdd(output + id_retinaPoint , val);
                atomicAdd(retinaDataInserted + id_retinaPoint , 1);
            }
        }
    }

    __global__ void RetinaTransform_FillRetinaAtomic (float * subImageDefs, 
                                                       float* input, int inputWidth, int inputHeight,
                                                       float* output,int outputDataSize,
                                                       float* retinaMask, int retinaDataSize, int retinaMaskColHint,
                                                       float* retinaDataInserted)
    {
        int id_pxl = blockDim.x * blockIdx.y * gridDim.x
				    + blockDim.x * blockIdx.x
				    + threadIdx.x;

		int2 subImg;
        int diameterPix;
        bool  safeBounds = 0;

        int x = id_pxl % inputWidth;
        int y = id_pxl/inputWidth;

        EstimateParForSubsample( subImageDefs,  safeBounds, inputWidth,  inputHeight,  subImg, diameterPix );

        if (id_pxl<inputWidth*inputHeight)
        {
            float minDist = 999999.9; // ??>? should be written bette
            int minIdx = 1;
            for (int id_retinaPoint=0 ; id_retinaPoint<retinaDataSize ; id_retinaPoint++)
            {
                float x_mask = (retinaMask[id_retinaPoint*retinaMaskColHint]*diameterPix);
                float y_mask = (retinaMask[id_retinaPoint*retinaMaskColHint+1]*diameterPix);

                x_mask += subImg.x;
                y_mask += subImg.y;

                float dist = (x-x_mask)*(x-x_mask) + (y-y_mask)*(y-y_mask);

                if (dist<minDist)
                {
                    minDist = dist;
                    minIdx  = id_retinaPoint;
                }
            }
            atomicAdd(output + minIdx , input[id_pxl]);
            atomicAdd(retinaDataInserted + minIdx , 1);
        }
    }



    

}


#define _SIZE_T_DEFINED 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __cplusplus 
#define __cplusplus 
#endif

#include "ColorScaleObserverSingle.cu"
#include "../Common/Statistics.cu"
#include "../Transforms/Transform2DKernels.cu"




extern "C"  
{

	//kernel code
    // There is ,,Vision/KMeansWM.cu/FocuserInputObserver''   that defines part of this function as device and has some features as plotting multiple lines 
	__global__ void FocuserInputObserver(float* values, float* pupilControl, int inputWidth, int inputHeight, unsigned int* pixels)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	
				+ blockDim.x*blockIdx.x				
				+ threadIdx.x;

		int numOfPixels = inputWidth * inputHeight;

		if(id < numOfPixels) //id of the thread is valid
		{		
			float cX = pupilControl[0]; // <-1, 1>
			float cY = pupilControl[1]; // <-1, 1>
			float subImgDiameter = pupilControl[2]; // <0,1>

			int maxDiameter = min(inputWidth, inputHeight);
			int diameterPix = (int)(subImgDiameter * maxDiameter);
			diameterPix = max(1, diameterPix);

			int cXPix = (int)(inputWidth * (cX + 1) * 0.5f);
			int cYPix = (int)(inputHeight * (cY + 1) * 0.5f);

			int subImgX = cXPix - diameterPix / 2;
			int subImgY = cYPix - diameterPix / 2;

			subImgX = max(subImgX, 0);
			subImgY = max(subImgY, 0);

			subImgX = min(subImgX, inputWidth - diameterPix);
			subImgY = min(subImgY, inputHeight - diameterPix);

			int px = id % inputWidth;
			int py = id / inputWidth;			

			float hue = 1.0f;
			float saturation = 0;				
			float value = values[id];			

			if (px >= subImgX && py >= subImgY && 
				px <= subImgX + diameterPix && py <= subImgY + diameterPix) 
			{
				saturation = 0.33f;
				value += 0.2f;
			}			

			if (px == cXPix || py == cYPix) 
			{
				saturation = 1.0f;
				value = 0.7f;
			}			

			value = fminf(fmaxf(value, 0), 1);

			pixels[id] = hsva_to_uint_rgba(hue, saturation, value, 1.0f);
		}
	}

	__constant__ int NUM_C_VALUES = 5;

	__global__ void PupilControlObserver(float* values, Centroid* centroids, int centroidsCount, int inputWidth, int inputHeight, unsigned int* pixels)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	
				+ blockDim.x*blockIdx.x				
				+ threadIdx.x;

		int numOfPixels = inputWidth * inputHeight;

		if(id < numOfPixels) //id of the thread is valid
		{			
			int px = id % inputWidth;
			int py = id / inputWidth;			

			float hue = 0.6f;
			float saturation = 0;				
			float value = values[id];			
			
			for (int i = 0; i < centroidsCount; i++) 
			{

				float cX = centroids[i].X; // <-1, 1>
				float cY = centroids[i].Y; // <-1, 1>		

				float cStdDevX = centroids[i].VarianceX;
				float cStdDevY = centroids[i].VarianceY;

				int cXPix = (int)(inputWidth * (cX + 1) * 0.5f);
				int cYPix = (int)(inputHeight * (cY + 1) * 0.5f);

				int cStdDevXPix = (int)(inputWidth * cStdDevX);
				int sStdDevYPix = (int)(inputHeight * cStdDevY);

				if (px >= cXPix - cStdDevXPix && py >= cYPix - sStdDevYPix && 
					px <= cXPix + cStdDevXPix && py <= cYPix + sStdDevYPix) 
				{
					hue = 0.33;
					saturation = 0.5;
					value += 0.2;
				}

				if (px >= cXPix - 2 && py >= cYPix - 2 && 
					px <= cXPix + 2 && py <= cYPix + 2) 
				{
					hue = 0.6;
					saturation = 1;
					value = 1;
				}					
			}
			
			value = fminf(fmaxf(value, 0), 1);

			pixels[id] = hsva_to_uint_rgba(hue, saturation, value, 1.0f);
		}
	}



    //------------------------------------------------------------------------------------------------------------------------
    //                          RETINA STUFF
    //------------------------------------------------------------------------------------------------------------------------

    __global__ void RetinaObserver_Mask(unsigned int* pixels, int pixelsWidth, int pixelsHeight, float* retinaPtsDefs, int retinaPtsDefsSize, float* subImageDefs)
    {
		int id = blockDim.x*blockIdx.y*gridDim.x	
				+ blockDim.x*blockIdx.x				
				+ threadIdx.x;
        float x,y;        

        int2 subImg;
        int diameterPix;
        bool  safeBounds = 0;

        EstimateParForSubsample( subImageDefs,  safeBounds,		pixelsWidth,  pixelsHeight,       subImg, diameterPix);

        if (id < retinaPtsDefsSize)
        {
            x = (float)subImg.x + (retinaPtsDefs[id * 2]*diameterPix);
            y = (float)subImg.y + (retinaPtsDefs[id * 2 + 1]*diameterPix);

            if (x>0 && y>0 && x<pixelsWidth && y<pixelsHeight)
                pixels[(int)x+(int)y*pixelsWidth] = GET_RGBA(0,255,0,255);
        }
    }


    // next code should be improved:
    //         - share memory for minDist
    //         - template specialization to have it just once.  It does not work now :(

    __global__ void RetinaObserver_UnMaskPatchFl(float* output2save, int pixelsWidth, int pixelsHeight, float* retinaPtsDefs, int retinaPtsDefsSize, float* retinaValues, float* subImageDefs)
    {
      int id_pxl = blockDim.x * blockIdx.y * gridDim.x
		          + blockDim.x * blockIdx.x
        	      + threadIdx.x;

		int2 subImg;
        int diameterPix;
        bool  safeBounds = 0;

        int x = id_pxl % pixelsWidth;
        int y = id_pxl / pixelsWidth;

        EstimateParForSubsample( subImageDefs,  safeBounds, pixelsWidth,  pixelsHeight,  subImg, diameterPix );

        if (id_pxl < pixelsWidth*pixelsHeight)
        {
            float minDist = 999999.9f;
            int minIdx = 0;
            for (int i = 0; i < retinaPtsDefsSize; i++)
            {
                float xr = subImg.x + retinaPtsDefs[i * 2]*(float)diameterPix;
                float yr = subImg.y + retinaPtsDefs[i * 2 + 1]*(float)diameterPix;
                float dist = (xr - x) * (xr - x) + (yr - y) * (yr - y);
                if (dist < minDist)
                {
                    minDist = dist;
                    output2save[id_pxl] = fminf(fmaxf(retinaValues[i],0.0f),1.0f);
                }
            }
        }
    }

    __global__ void RetinaObserver_UnMaskPatchVBO(unsigned int* pixels, int pixelsWidth, int pixelsHeight, float* retinaPtsDefs, int retinaPtsDefsSize,  float* retinaValues, float* subImageDefs)
    {
      int id_pxl = blockDim.x * blockIdx.y * gridDim.x
		          + blockDim.x * blockIdx.x
        	      + threadIdx.x;

		int2 subImg;
        int diameterPix;
        bool  safeBounds = 0;

        int x = id_pxl % pixelsWidth;
        int y = id_pxl / pixelsWidth;

        EstimateParForSubsample( subImageDefs,  safeBounds, pixelsWidth,  pixelsHeight,  subImg, diameterPix );

        if (id_pxl < pixelsWidth*pixelsHeight)
        {
            float minDist = 999999.9f;
            int minIdx = 0;
            for (int i = 0; i < retinaPtsDefsSize; i++)
            {
                float xr = subImg.x + retinaPtsDefs[i * 2]*(float)diameterPix;
                float yr = subImg.y + retinaPtsDefs[i * 2 + 1]*(float)diameterPix;
                float dist = (xr - x) * (xr - x) + (yr - y) * (yr - y);
                if (dist < minDist)
                {
                    minDist = dist;
                    pixels[id_pxl] = hsva_to_uint_rgba(0.5f, 1.0f, fminf(fmaxf(retinaValues[i],0.0f),1.0f), 1.0f);
                }
            }
        }
    }



}


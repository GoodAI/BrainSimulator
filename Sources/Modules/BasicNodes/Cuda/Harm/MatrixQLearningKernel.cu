//Includes for IntelliSense 
#define _SIZE_T_DEFINED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

#include "../Observers/ColorHelpers.cu"

// @author jv
extern "C"  
{	
	// detects multiple max values
	__global__ void findMaxIndMultipleDetector(float *input, int* maxInd, int size)
	{
		int maxIndex = 0;
		int count = 1;

		for (int i = 1; i < size; i++){
			if (input[maxIndex] < input[i]){
				maxIndex = i;
				count = 1;
			}
			else if (input[maxIndex] == input[i]){
				count++;
			}
		}
		if(count>1)
			maxInd[0] = -1;
		else
			maxInd[0] = maxIndex;
	}

	__global__ void oneOfNSelection(float *buffer, int* index, int size, float value)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x						//blocks preceeding current block
			+ threadIdx.x;

		if (threadId < size && threadId != index[0])
		{
			buffer[threadId] = 0;

		}
		else if (threadId < size && threadId == index[0]){
			buffer[threadId] = value;
		}
	}

	__global__ void dummyKernel()
	{
	}

	__global__ void copyKernel(float* from, float* to, int size)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x
				+ blockDim.x*blockIdx.x					
				+ threadIdx.x;

		if(threadId < size)
		{
			to[threadId] = from[threadId];
		}
	}

	__global__ void detectChanges(float* a, float* b, float* result, int size, float value)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x
				+ blockDim.x*blockIdx.x					
				+ threadIdx.x;

		if(threadId < size)
		{
			if(a[threadId] > b[threadId])
			{
				result[threadId] = value;
			}
			else if(a[threadId] <b[threadId])
			{
				result[threadId] = -value;
			}
			else
			{
				result[threadId] = 0;
			}
		}
	}

	__global__ void createTexture(
		float* plotValues, int* actionIndices, unsigned int* actionLabels, int numOfActions, 
		int patchWidth, int patchHeight, 
		float minValue, float maxValue, int itemsX, int itemsY, unsigned int* pixels) 
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;	
		
		int textureWidth = patchWidth * itemsX;
		int textureHeight = patchHeight * itemsY;
		int size = textureWidth * textureHeight;

		int patchX = (threadId / patchWidth) % itemsX;
		int patchY = threadId / (patchWidth * patchHeight * itemsX);

		int pixX = threadId % patchWidth;
		int pixY = threadId / textureWidth % patchHeight;

		int actionIndex = actionIndices[patchX * itemsY + (itemsY - patchY - 1)];
		// if the utility value is zero, there is no action (marged by black/zeros)
		unsigned int noActionFlag = (plotValues[patchX * itemsY + (itemsY - patchY - 1)] != 0) ? 0xFFFFFFFF : 0;

		if (threadId < size) 
		{						
			pixels[threadId] = float_to_uint_rgba(plotValues[patchX * itemsY + (itemsY - patchY - 1)], 2, 2, minValue, maxValue);	
			pixels[threadId] &= noActionFlag;
			pixels[threadId] |= actionLabels[pixY * numOfActions * patchWidth + actionIndex * patchWidth + pixX];
		}		
	}

	__global__ void crate3Dplot(float* plotValues, float patchSize, int itemsX, int itemsY, float maxValue, float* vertexData) 
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;	

		int size = itemsX * itemsY;		
		float texSizeX = 1.0f / itemsX;
		float texSizeY = 1.0f / itemsY;

		int patchX = threadId / itemsY;
		int patchY = itemsY - (threadId % itemsY) - 1;

		if (threadId < size) 
		{
			float height = plotValues[threadId] / maxValue;
			float gap = 0.000;

			float3* vertTop = (float3*)vertexData;			
			float2* texCoords = (float2*)(vertexData + (60 * size));			

			//top side
			vertTop[threadId * 4].x = patchX * patchSize + gap;			
			vertTop[threadId * 4].y = height;
			vertTop[threadId * 4].z = patchY * patchSize + gap;

			texCoords[threadId * 4].x = patchX * texSizeX;
			texCoords[threadId * 4].y = patchY * texSizeY;

			vertTop[threadId * 4 + 1].x = (patchX + 1) * patchSize - gap;			
			vertTop[threadId * 4 + 1].y = height;
			vertTop[threadId * 4 + 1].z = patchY * patchSize + gap;

			texCoords[threadId * 4 + 1].x = (patchX + 1) * texSizeX;
			texCoords[threadId * 4 + 1].y = patchY * texSizeY;

			vertTop[threadId * 4 + 2].x = (patchX + 1) * patchSize - gap;			
			vertTop[threadId * 4 + 2].y = height;
			vertTop[threadId * 4 + 2].z = (patchY + 1) * patchSize - gap;

			texCoords[threadId * 4 + 2].x = (patchX + 1) * texSizeX;
			texCoords[threadId * 4 + 2].y = (patchY + 1) * texSizeY;

			vertTop[threadId * 4 + 3].x = patchX * patchSize + gap;			
			vertTop[threadId * 4 + 3].y = height;
			vertTop[threadId * 4 + 3].z = (patchY + 1) * patchSize - gap;

			texCoords[threadId * 4 + 3].x = patchX * texSizeX;
			texCoords[threadId * 4 + 3].y = (patchY + 1) * texSizeY;

			float3* vertLeft = (float3*)(vertexData + 12 * size);

			//left side
			vertLeft[threadId * 4] = vertTop[threadId * 4];
			vertLeft[threadId * 4].y = 0;
			vertLeft[threadId * 4 + 1] = vertTop[threadId * 4];			

			vertLeft[threadId * 4 + 2] = vertTop[threadId * 4 + 3];
			vertLeft[threadId * 4 + 3] = vertTop[threadId * 4 + 3];
			vertLeft[threadId * 4 + 3].y = 0;

			float3* vertFar = (float3*)(vertexData + 24 * size);

			//far side
			vertFar[threadId * 4] = vertTop[threadId * 4 + 2];
			vertFar[threadId * 4].y = 0;
			vertFar[threadId * 4 + 1] = vertTop[threadId * 4 + 3];			
			vertFar[threadId * 4 + 1].y = 0;

			vertFar[threadId * 4 + 2] = vertTop[threadId * 4 + 3];
			vertFar[threadId * 4 + 3] = vertTop[threadId * 4 + 2];			

			float3* vertNear = (float3*)(vertexData + 36 * size);

			//near side
			vertNear[threadId * 4] = vertTop[threadId * 4 + 1];
			vertNear[threadId * 4].y = 0;
			vertNear[threadId * 4 + 1] = vertTop[threadId * 4];			
			vertNear[threadId * 4 + 1].y = 0;

			vertNear[threadId * 4 + 2] = vertTop[threadId * 4];
			vertNear[threadId * 4 + 3] = vertTop[threadId * 4 + 1];			

			float3* vertRight = (float3*)(vertexData + 48 * size);

			//right side
			vertRight[threadId * 4] = vertTop[threadId * 4 + 2];
			vertRight[threadId * 4].y = 0;
			vertRight[threadId * 4 + 1] = vertTop[threadId * 4 + 2];			

			vertRight[threadId * 4 + 2] = vertTop[threadId * 4 + 1];
			vertRight[threadId * 4 + 3] = vertTop[threadId * 4 + 1];
			vertRight[threadId * 4 + 3].y = 0;
		}
	}
}

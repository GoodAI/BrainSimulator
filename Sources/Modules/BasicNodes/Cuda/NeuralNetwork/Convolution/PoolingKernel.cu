//Includes for IntelliSense 
#define _SIZE_T_DEFINED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math.h>
#include "..\Activation\ActivationFunction.cu"

extern "C"
{

	__device__ int indexFromXY (int x, int y, int width)
	{
		return y * width + x;
	}

	__global__ void PoolingForwardKernel (
		float *inputPtr,
		float *outputPtr,
		int *activatedNeuronsPtr,
		int inputWidth, int inputSize,
		int filterWidth, int filterHeight,
		int horStride, int verStride,
		int outputWidth, int outputSize,
		int thisLayerSize
	)
	{
		int idx = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
				+ blockDim.x * blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if (idx < thisLayerSize)
		{

			int depth = idx / outputSize;
			int depthShift = depth * inputSize;


			int inputTileX = (idx % outputSize) % outputWidth;
			int inputTileY = (idx % outputSize) / outputWidth;
			

			int y = inputTileY * verStride;
			int maxY = y;

			int maxX = inputTileX * horStride;

			
			float maxValue = inputPtr[depthShift + indexFromXY(maxX, y, inputWidth)];

			for (int j = 0; j < filterHeight; j++)
			{
				int x = inputTileX * horStride;
				for (int i = 0; i < filterWidth; i++)
				{
					float value = inputPtr[depthShift + indexFromXY(x, y, inputWidth)];
					if (value > maxValue) {
						value = maxValue;
						maxX = x;
						maxY = y;
					}
					++x;
				}
				++y;
			}

			// probably no need to save weighted input (without activation), since pool layer has no activation
			// if it would have, it would be needed to add this
			outputPtr[idx] = maxValue;
			activatedNeuronsPtr[idx] = depthShift + indexFromXY(maxX, maxY, inputWidth);
		}
	}


	__global__ void PoolingBackwardKernel (
		ActivationFunctionEnum inputActFunc,
		float *thisLayerDelta,
		float *inputLayerDelta,
		float *inputWeightedPtr,
		int *activatedNeuronsPtr,
		int thisLayerSize
	)
	{
		int idx = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
				+ blockDim.x * blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if (idx < thisLayerSize)
		{
			int inputIdx = activatedNeuronsPtr[idx];
			inputLayerDelta[inputIdx] += thisLayerDelta[idx] * EvaluateDerivative(inputActFunc, inputWeightedPtr[inputIdx]);
		}
	}
}
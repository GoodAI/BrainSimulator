//Includes for IntelliSense 
#define _SIZE_T_DEFINED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math.h>

extern "C"
{

	__device__ int indexFromXY(int x, int y, int width)
	{
		return y * width + x;
	}

	__global__ void ConvolutionForwardKernel(
		float *inputPtr,
		float *filterPtr,
		float *biasPtr,
		float *outputPtr,
		int filterWidth, int filterHeight,
		int filterDepth,
		int filterSliceSize, // one layer of filter volume, fW * fH
		int filterSize, // one filter volume, fW * fH * inputDepth
		int inputSliceSize, // one layer of input data, e.g. one channel of an RGB image
		int inputWidth,
		int outputSize, // size of one resulting output layer = one learned filter, oW * oH (there are filterCount of these)
		int filtersPerRow,
		int horStride, int verStride,
		int thisLayerSize
		)
	{
		int idx = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (idx < thisLayerSize)
		{
			int filterIdx = idx / outputSize;
			
			int inputTileX = (idx % outputSize) % filtersPerRow;
			int inputTileY = (idx % outputSize) / filtersPerRow;

			
			float result = 0;

			for (size_t z = 0; z < filterDepth; z++) // Z
			{
				int inputIndex = z * inputSliceSize;
				int y = inputTileY * verStride;

				for (size_t j = 0; j < filterHeight; j++) // Y
				{
					int x = inputTileX * horStride;
					int filterIndex = filterSize * filterIdx + z * filterSliceSize;

					for (size_t i = 0; i < filterWidth; i++) // X
					{
						result +=
							inputPtr[inputIndex + indexFromXY(x, y, inputWidth)] * // input
							filterPtr[filterIndex + indexFromXY(i, j, filterWidth)]; // weight
						++x;
					}
					++y;

				}
			}

			result += biasPtr[filterIdx];

			outputPtr[idx] = result;

		}
	}


	// computes deltas
	__global__ void ConvolutionBackwardKernel(
		float *filterPtr,
		float *deltaPtr,
		float *nextLayerDeltaPtr,
		int filterWidth, int filterHeight,
		int filterDepth,
		int filterSliceSize, // one layer of filter volume, fW * fH
		int filterSize, // one filter volume, fW * fH * inputDepth
		int inputSliceSize, // one layer of input data, e.g. one channel of an RGB image
		int inputWidth,
		int outputSize, // size of one resulting output layer = one learned filter, oW * oH (there are filterCount of these)
		int filtersPerRow,
		int horStride, int verStride,
		int thisLayerSize
		)
	{
		int idx = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (idx < thisLayerSize)
		{
			int filterIdx = idx / outputSize;

			int inputTileX = (idx % outputSize) % filtersPerRow;
			int inputTileY = (idx % outputSize) / filtersPerRow;


			float result = 0;

			for (size_t z = 0; z < filterDepth; z++) // Z
			{
				int y = inputTileY * verStride;
				for (size_t j = 0; j < filterHeight; j++) // Y
				{
					int x = inputTileX * horStride;
					for (size_t i = 0; i < filterWidth; i++) // X
					{
						result +=
							filterPtr[filterSize * filterIdx + z * filterSliceSize + indexFromXY(filterWidth - 1 - i, filterHeight - 1 - j, filterWidth)] *
							nextLayerDeltaPtr[z * inputSliceSize + indexFromXY(x, y, inputWidth)];
						++x;
					}
					++y;
				}
			}

			deltaPtr[idx] = result;

		}
	}

	__global__ void PadImageKernel(
		float *inputPtr,
		float *outputPtr,
		int inputWidth,
		int pad,
		int inputSize, // one depth slice / one layer / one color channel
		int outputSize,
		int totalInputSize // whole image (all color channels combined)
		)
	{
		int idx = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (idx < totalInputSize)
		{
			int depth = idx / inputSize;

			int outputDepthShift = depth * outputSize;

			int rowIdx = (idx % inputSize) / inputWidth;
			int colIdx = (idx % inputSize) % inputWidth;

			outputPtr[indexFromXY(pad + colIdx, pad + rowIdx, pad + inputWidth + pad) + outputDepthShift] = inputPtr[idx];
		}
	}
}
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
#include <stdio.h>

extern "C"
{

	__device__ int indexFromXY(int x, int y, int width)
	{
		return y * width + x;
	}

	__global__ void ConvolutionForwardKernel(
		ActivationFunctionEnum activationFunction,
		float *inputPtr,
		float *filterPtr,
		float *biasPtr,
		float *outputPtr,
		float *outputWeightedPtr,
		int filterWidth, int filterHeight, int filterDepth,
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

			outputWeightedPtr[idx] = result;
			outputPtr[idx] = Evaluate(activationFunction, result);

		}
	}


	// computes deltas
	// launched size(prevDeltaPtr) times, i.e. separately for each delta to be computed
	__global__ void ConvolutionBackwardKernel(
		ActivationFunctionEnum inputActFunc,
		float *filterPtr,
		float *thisDeltaPtr,
		float *inputDeltaPtr,
		float *inputWeightedPtr,
		int filterCount,
		int inputSliceSize, // one layer of input data, e.g. one channel of an RGB image
		int inputPaddedSliceSize, // same, but accounting for possible padding
		int padding,
		int inputWidth, int inputHeight,
		int filterWidth, int filterHeight,
		int filterSliceSize, // one layer of filter volume, fW * fH
		int outputWidth, int outputSliceSize, // size of one resulting output layer = one learned filter, oW * oH (there are filterCount of these)
		int horStride, int verStride,
		int prevLayerSize
		)
	{
		int idx = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (idx < prevLayerSize)
		{
			float delta = 0;
			for (size_t filterIdx = 0; filterIdx < filterCount; filterIdx++)
			{
				int currentDepth = idx / inputSliceSize; // currentZ

				int weightDepthShift = currentDepth * filterSliceSize;
				int deltaDepthShift = currentDepth * outputSliceSize;

				// index in the current slice (ignoring depth), accounting for padding
				int rowIdx = (idx % inputSliceSize) / inputWidth;
				int currentIdx = (idx % inputSliceSize) + (2 * padding * padding) + (padding * inputWidth) + padding + (padding * padding * rowIdx);

				int paddedWidth = padding + inputWidth + padding;
				int paddedHeight = padding + inputHeight + padding;
				int currentX = currentIdx % paddedWidth;
				int currentY = currentIdx / paddedWidth;

				int filterX = 0;
				int filterY = 0;
				// cycle filter through the whole (virtually padded) image
				for (int j = 0; filterY + filterHeight <= paddedHeight; j++, filterY += verStride)
				{
					for (int i = 0; filterX + filterWidth <= paddedWidth; i++, filterX += horStride)
					{
						if ( // check if the current neuron is in the filter's receptive field
							filterX <= currentX && filterX + filterWidth > currentX &&
							filterY <= currentY && filterY + filterHeight > currentY)
						{
							// identify the proper filter part (weight)
							int weightIdx = weightDepthShift + indexFromXY(currentX - filterX, currentY - filterY, filterWidth);
							// identify the proper output neuron (delta)
							int deltaIdx = deltaDepthShift + j * outputWidth + i;
							delta += filterPtr[weightIdx] * thisDeltaPtr[deltaIdx];
						}
					}
				}
			}

			delta *= EvaluateDerivative(inputActFunc, inputWeightedPtr[idx]);

			// NO batch learning -> might this be a problem?
			inputDeltaPtr[idx] = delta;

		}
	}



	__global__ void ConvolutionSGDUpdateWeightsKernel(
		float learningRate, float momentum,
		float *filterPtr,
		float *biasPtr, float *previousBiasDeltaPtr,
		float *thisDeltaPtr, float *previousWeightDeltaPtr,
		float *inputPaddedPtr,
		int inputPaddedWidth, int inputPaddedSliceSize, // needs to account for padding!
		int filterWidth,
		int filterSliceSize, // one layer of filter volume, fW * fH
		int filterSize,
		int outputWidth, int outputHeight, int outputSliceSize, // size of one resulting output layer = one learned filter, oW * oH (there are filterCount of these)
		int horStride, int verStride, //float *outputPtr,
		float L1Lambda, float L2Lambda,
		int weightCount // == filterSize * filterCount
		)
	{
		int idx = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (idx < weightCount)
		{
			// determine the exact weight to be updated (one thread corresponds to exactly one weight)
			// index of the weight inside the filter:
			int filterX = (idx % filterSliceSize) % filterWidth;
			int filterY = (idx % filterSliceSize) / filterWidth;
			// filterZ:
			int inputDepth = (idx % filterSize) / filterSliceSize;
			int outputDepth = idx / filterSize; // index of the current filter

			int inputDepthShift = inputDepth * inputPaddedSliceSize;
			int outputDepthShift = outputDepth * outputSliceSize;
			int filterInputShift = filterX + filterY * inputPaddedWidth; // by how much is the current weight shifted from the upper-left corner of the filter IN THE INPUT IMAGE

			// apply the filter over the whole image (do convolution again) with this one weight
			float delta = 0;
			float biasDelta = 0;
			//int a = 0;
			for (size_t j = 0; j < outputHeight; j++)
			{
				for (size_t i = 0; i < outputWidth; i++)
				{
					delta += thisDeltaPtr[outputDepthShift + i + j * outputWidth] *
						inputPaddedPtr[
							inputDepthShift +
								j * verStride * inputPaddedWidth +
								i * horStride +
								filterInputShift
						];

					// update bias (one bias per filter, so only do it if we are in the first weight of any filter)
					// it seems to work better without the following condition though it shouldn't be the case
					if (idx % filterSize == 0)
						biasDelta += thisDeltaPtr[outputDepthShift + i + j * outputWidth];

				}
			}

			// UPDATE BIAS --------------------------------
			if (idx % filterSize == 0)
			{
				// bias usually doesn't get regularised
				//biasDelta += L1Lambda * sign(biasPtr[idx / filterSize]) + L2Lambda * biasPtr[idx / filterSize];
				biasDelta *= learningRate;

				if (momentum != 0)
				{
					biasDelta += momentum * previousBiasDeltaPtr[idx / filterSize];
					previousBiasDeltaPtr[idx / filterSize] = biasDelta;
				}

				biasPtr[idx / filterSize] -= biasDelta;
			}
			// -------------------------------------------


			// UPDATE WEIGHT -----------------------------

			// add regularization
			delta += L1Lambda * sign(filterPtr[idx]) + L2Lambda * filterPtr[idx];
			delta *= learningRate;

			// add momentum
			if (momentum != 0)
			{
				delta += momentum * previousWeightDeltaPtr[idx];
				previousWeightDeltaPtr[idx] = delta;
			}

			if (delta != 0)
			{
				filterPtr[idx] -= delta;
			}
			// -----------------------------------------------
		}
	}





	__global__ void ConvolutionRMSPropUpdateWeightsKernel(
		float learningRate, float momentum,
		float *filterPtr,
		float *biasPtr, float *previousBiasDeltaPtr,
		float *thisDeltaPtr, float *previousWeightDeltaPtr,
		float *inputPaddedPtr,
		int inputPaddedWidth, int inputPaddedSliceSize, // needs to account for padding!
		int filterWidth,
		int filterSliceSize, // one layer of filter volume, fW * fH
		int filterSize,
		int outputWidth, int outputHeight, int outputSliceSize, // size of one resulting output layer = one learned filter, oW * oH (there are filterCount of these)
		int horStride, int verStride, //float *outputPtr,
		float L1Lambda, float L2Lambda,
		float *meanSquareWeight, float *meanSquareBias, float smoothingFactor,
		int weightCount // == filterSize * filterCount
		)
	{
		int idx = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (idx < weightCount)
		{
			// determine the exact weight to be updated (one thread corresponds to exactly one weight)
			// index of the weight inside the filter:
			int filterX = (idx % filterSliceSize) % filterWidth;
			int filterY = (idx % filterSliceSize) / filterWidth;
			// filterZ:
			int inputDepth = (idx % filterSize) / filterSliceSize;
			int outputDepth = idx / filterSize; // index of the current filter

			int inputDepthShift = inputDepth * inputPaddedSliceSize;
			int outputDepthShift = outputDepth * outputSliceSize;
			int filterInputShift = filterX + filterY * inputPaddedWidth; // by how much is the current weight shifted from the upper-left corner of the filter IN THE INPUT IMAGE

			// apply the filter over the whole image (do convolution again) with this one weight
			float delta = 0;
			float biasDelta = 0;
			for (size_t j = 0; j < outputHeight; j++)
			{
				for (size_t i = 0; i < outputWidth; i++)
				{
					delta += thisDeltaPtr[outputDepthShift + i + j * outputWidth] *
						inputPaddedPtr[
							inputDepthShift +
								j * verStride * inputPaddedWidth +
								i * horStride +
								filterInputShift
						];

					// update bias (one bias per filter, so only do it if we are in the first weight of any filter)
					// it seems to work better without the following condition though it shouldn't be the case
					if (idx % filterSize == 0)
						biasDelta += thisDeltaPtr[outputDepthShift + i + j * outputWidth];

				}
			}

			// UPDATE BIAS ---------------------------
			if (idx % filterSize == 0)
			{
				// bias usually doesn't get regularised
				//biasDelta += L1Lambda * sign(biasPtr[idx / filterSize]) + L2Lambda * biasPtr[idx / filterSize];

				if (momentum != 0)
				{
					biasDelta += momentum * previousBiasDeltaPtr[idx / filterSize];
					previousBiasDeltaPtr[idx / filterSize] = biasDelta;
				}
				// calculate meansquare
				meanSquareBias[idx / filterSize] = smoothingFactor * meanSquareBias[idx / filterSize] + (1.0f - smoothingFactor) * biasDelta * biasDelta;
				if (meanSquareBias[idx / filterSize] != 0)
					biasDelta /= sqrtf(meanSquareBias[idx / filterSize]);

				biasPtr[idx / filterSize] -= learningRate * biasDelta;
			}
			// ----------------------------------------


			// UPDATE WEIGHT --------------------------

			// add regularization
			delta += L1Lambda * sign(filterPtr[idx]) + L2Lambda * filterPtr[idx];
			// add momentum
			if (momentum != 0)
			{
				delta += momentum * previousWeightDeltaPtr[idx];
				previousWeightDeltaPtr[idx] = delta;
			}

			// calculate meansquare
			meanSquareWeight[idx] = smoothingFactor * meanSquareWeight[idx] + (1.0f - smoothingFactor) * delta * delta;
			if (meanSquareWeight[idx] != 0)
				delta /= sqrtf(meanSquareWeight[idx]);

			if (delta != 0)
			{
				filterPtr[idx] -= learningRate * delta;
			}
			// -----------------------------------------
		}
	}




	__global__ void ConvolutionAdadeltaUpdateWeightsKernel(
		float *filterPtr,
		float *biasPtr, float *previousBiasDeltaPtr,
		float *thisDeltaPtr, float *previousWeightDeltaPtr,
		float *inputPaddedPtr,
		int inputPaddedWidth, int inputPaddedSliceSize, // needs to account for padding!
		int filterWidth,
		int filterSliceSize, // one layer of filter volume, fW * fH
		int filterSize,
		int outputWidth, int outputHeight, int outputSliceSize, // size of one resulting output layer = one learned filter, oW * oH (there are filterCount of these)
		int horStride, int verStride, //float *outputPtr,
		float L1Lambda, float L2Lambda,
		float *adaSquares, float *adaDeltas, float *adaBiasSquares, float *adaBiasDeltas, float ro, float epsilon,
		int weightCount // == filterSize * filterCount
		)
	{
		int idx = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (idx < weightCount)
		{
			// determine the exact weight to be updated (one thread corresponds to exactly one weight)
			// index of the weight inside the filter:
			int filterX = (idx % filterSliceSize) % filterWidth;
			int filterY = (idx % filterSliceSize) / filterWidth;
			// filterZ:
			int inputDepth = (idx % filterSize) / filterSliceSize;
			int outputDepth = idx / filterSize; // index of the current filter

			int inputDepthShift = inputDepth * inputPaddedSliceSize;
			int outputDepthShift = outputDepth * outputSliceSize;
			int filterInputShift = filterX + filterY * inputPaddedWidth; // by how much is the current weight shifted from the upper-left corner of the filter IN THE INPUT IMAGE

			// apply the filter over the whole image (do convolution again) with this one weight
			float delta = 0;
			float biasDelta = 0;

			for (size_t j = 0; j < outputHeight; j++)
			{
				for (size_t i = 0; i < outputWidth; i++)
				{
					delta += thisDeltaPtr[outputDepthShift + i + j * outputWidth] *
						inputPaddedPtr[
							inputDepthShift +
								j * verStride * inputPaddedWidth +
								i * horStride +
								filterInputShift
						];

					// update bias (one bias per filter, so only do it if we are in the first weight of any filter)
					// it seems to work better without the following condition though it shouldn't be the case
					if (idx % filterSize == 0)
						biasDelta += thisDeltaPtr[outputDepthShift + i + j * outputWidth];
				}
			}

			// UPDATE WEIGHT --------------------------

			// should we even support regularization here? ok...
			delta += L1Lambda * sign(filterPtr[idx]) + L2Lambda * filterPtr[idx];


			// adadelta:
			adaSquares[idx] = ro * adaSquares[idx] + (1 - ro) * delta * delta;
			float dx = -sqrtf((adaDeltas[idx] + epsilon) / (adaSquares[idx] + epsilon)) * delta;
			adaDeltas[idx] = ro * adaDeltas[idx] + (1 - ro) * dx * dx;
			filterPtr[idx] += dx;  // there should be a '+' here, but '+' doesn't and '-' does work...?

			// -----------------------------------------


			// UPDATE BIAS ---------------------------
			if (idx % filterSize == 0)
			{
				// bias usually doesn't get regularised
				//biasDelta += L1Lambda * sign(biasPtr[idx / filterSize]) + L2Lambda * biasPtr[idx / filterSize];

				adaBiasSquares[idx] = ro * adaBiasSquares[idx] + (1 - ro) * biasDelta * biasDelta;
				float dx = -sqrtf((adaBiasDeltas[idx] + epsilon) / (adaBiasSquares[idx] + epsilon)) * biasDelta;
				adaBiasDeltas[idx] = ro * adaBiasDeltas[idx] + (1 - ro) * dx * dx;
				filterPtr[idx] += dx;  // there should be a '+' here, but '+' doesn't and '-' does work...?
			}
			// ----------------------------------------

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

			int rowIdx = (idx % inputSize) / inputWidth;
			int colIdx = (idx % inputSize) % inputWidth;

			outputPtr[indexFromXY(pad + colIdx, pad + rowIdx, pad + inputWidth + pad) + (depth * outputSize)] = inputPtr[idx];
		}
	}
}
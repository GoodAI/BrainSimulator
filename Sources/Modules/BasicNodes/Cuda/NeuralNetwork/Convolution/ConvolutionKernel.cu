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
		int filterSize, // one filter volume, fW * fH * inputDepth
		int outputWidth, int outputHeight, int outputSliceSize, // size of one resulting output layer = one learned filter, oW * oH (there are filterCount of these)
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

				int inputDepth = idx / inputSliceSize; // currentZ

				// shift to correct filter index by current index and size, then shift inside the correct filter to the correct depth
				int filterDepthShift = filterIdx * filterSize + inputDepth * filterSliceSize;

				int deltaDepthShift = filterIdx * outputSliceSize;

				// index in the current slice (ignoring depth), accounting for padding
				int rowIdx = (idx % inputSliceSize) / inputWidth;
				int currentIdx = (idx % inputSliceSize) + (2 * padding * padding) + (padding * inputWidth) + padding + (padding * padding * rowIdx);

				int paddedWidth = padding + inputWidth + padding;
				int paddedHeight = padding + inputHeight + padding;
				int currentX = currentIdx % paddedWidth;
				int currentY = currentIdx / paddedWidth;

				int filterY = 0;
				// cycle filter through the whole (virtually padded) image
				for (int j = 0; j < outputHeight; j++)
				{
					// check if the current neuron is in the filter's vertical receptive field
					if (filterY <= currentY && currentY < filterY + filterHeight) {


						int filterX = 0;
						for (int i = 0; i < outputWidth; i++)
						{
							// check if the current neuron is in the filter's horizontal receptive field
							if (filterX <= currentX && currentX < filterX + filterWidth)
							{
								// identify the proper filter part (weight)
								int filterIdx = filterDepthShift + indexFromXY(currentX - filterX, currentY - filterY, filterWidth);
								// identify the proper output neuron (delta)
								int deltaIdx = deltaDepthShift + indexFromXY(i, j, outputWidth);
								delta += filterPtr[filterIdx] * thisDeltaPtr[deltaIdx];
							}
							filterX += horStride;
						}


					}
					filterY += verStride;
				}
			}

			delta *= EvaluateDerivative(inputActFunc, inputWeightedPtr[idx]);

			inputDeltaPtr[idx] += delta;

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
		int batchSize,
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
			float gradient = 0;
			float biasGradient = 0;
			for (size_t j = 0; j < outputHeight; j++)
			{
				for (size_t i = 0; i < outputWidth; i++)
				{
					gradient += thisDeltaPtr[outputDepthShift + i + j * outputWidth] *
						inputPaddedPtr[
							inputDepthShift +
								j * verStride * inputPaddedWidth +
								i * horStride +
								filterInputShift
						];

					/*if (idx == 49)
					{
						thisDeltaPtr[outputDepthShift + i + j * outputWidth] = -100;
						inputPaddedPtr[
							inputDepthShift +
								j * verStride * inputPaddedWidth +
								i * horStride +
								filterInputShift
						] = -100;
					}*/

					// update bias (one bias per filter, so only do it if we are in the first weight of any filter)
					// it seems to work better without the following condition though it shouldn't be the case
					if (idx % filterSize == 0)
						biasGradient += thisDeltaPtr[outputDepthShift + i + j * outputWidth];

				}
			}


			// UPDATE WEIGHT -----------------------------

			// add regularization
			gradient += L1Lambda * sign(filterPtr[idx]) + L2Lambda * filterPtr[idx];

			float dx = -gradient * learningRate / batchSize;

			// add momentum
			if (momentum != 0)
			{
				dx += momentum * previousWeightDeltaPtr[idx];
				previousWeightDeltaPtr[idx] = dx;
			}

			filterPtr[idx] += dx;
			// -----------------------------------------------

			// UPDATE BIAS --------------------------------
			if (idx % filterSize == 0)
			{
				// bias usually doesn't get regularised
				// biasDelta += L1Lambda * sign(biasPtr[idx / filterSize]) + L2Lambda * biasPtr[idx / filterSize];
				float dx = -biasGradient * learningRate / batchSize;

				if (momentum != 0)
				{
					dx += momentum * previousBiasDeltaPtr[idx / filterSize];
					previousBiasDeltaPtr[idx / filterSize] = dx;
				}

				biasPtr[idx / filterSize] += dx;
			}
			// -------------------------------------------

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
		int batchSize,
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
			float gradient = 0;
			float biasGradient = 0;
			for (size_t j = 0; j < outputHeight; j++)
			{
				for (size_t i = 0; i < outputWidth; i++)
				{
					gradient += thisDeltaPtr[outputDepthShift + i + j * outputWidth] *
						inputPaddedPtr[
							inputDepthShift +
								j * verStride * inputPaddedWidth +
								i * horStride +
								filterInputShift
						];

					// update bias (one bias per filter, so only do it if we are in the first weight of any filter)
					// it seems to work better without the following condition though it shouldn't be the case
					if (idx % filterSize == 0)
						biasGradient += thisDeltaPtr[outputDepthShift + i + j * outputWidth];

				}
			}


			// UPDATE WEIGHT -----------------------------
			// add regularization
			gradient += L1Lambda * sign(filterPtr[idx]) + L2Lambda * filterPtr[idx];
			gradient /= batchSize;

			// calculate meansquare
			meanSquareWeight[idx] = smoothingFactor * meanSquareWeight[idx] + (1.0f - smoothingFactor) * gradient * gradient;
			if (meanSquareWeight[idx] != 0)
				gradient /= sqrtf(meanSquareWeight[idx]);

			float dx = -gradient * learningRate;

			// add momentum
			if (momentum != 0)
			{
				dx += momentum * previousWeightDeltaPtr[idx];
				previousWeightDeltaPtr[idx] = dx;
			}

			filterPtr[idx] += dx;
			// -----------------------------------------


			// UPDATE BIAS ---------------------------
			if (idx % filterSize == 0)
			{
				// bias usually doesn't get regularised
				//biasDelta += L1Lambda * sign(biasPtr[idx / filterSize]) + L2Lambda * biasPtr[idx / filterSize];
				biasGradient /= batchSize;

				// calculate meansquare
				meanSquareBias[idx / filterSize] = smoothingFactor * meanSquareBias[idx / filterSize] + (1.0f - smoothingFactor) * biasGradient * biasGradient;
				if (meanSquareBias[idx / filterSize] != 0)
					biasGradient /= sqrtf(meanSquareBias[idx / filterSize]);

				float dx = -biasGradient * learningRate;

				if (momentum != 0)
				{
					dx += momentum * previousBiasDeltaPtr[idx / filterSize];
					previousBiasDeltaPtr[idx / filterSize] = dx;
				}

				biasPtr[idx / filterSize] += dx;
			}
			// ----------------------------------------



		}
	}




	__global__ void ConvolutionAdadeltaUpdateWeightsKernel(
		float *filterPtr,
		float *biasPtr,
		float *thisDeltaPtr,
		float *inputPaddedPtr,
		int inputPaddedWidth, int inputPaddedSliceSize, // needs to account for padding!
		int filterWidth,
		int filterSliceSize, // one layer of filter volume, fW * fH
		int filterSize,
		int outputWidth, int outputHeight, int outputSliceSize, // size of one resulting output layer = one learned filter, oW * oH (there are filterCount of these)
		int horStride, int verStride, //float *outputPtr,
		float L1Lambda, float L2Lambda,
		float *adaSquares, float *adaDeltas, float *adaBiasSquares, float *adaBiasDeltas, float ro, float epsilon,
		int batchSize,
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
			float gradient = 0;
			float biasGradient = 0;
			for (size_t j = 0; j < outputHeight; j++)
			{
				for (size_t i = 0; i < outputWidth; i++)
				{
					gradient += thisDeltaPtr[outputDepthShift + i + j * outputWidth] *
						inputPaddedPtr[
							inputDepthShift +
								j * verStride * inputPaddedWidth +
								i * horStride +
								filterInputShift
						];

					// update bias (one bias per filter, so only do it if we are in the first weight of any filter)
					// it seems to work better without the following condition though it shouldn't be the case
					if (idx % filterSize == 0)
						biasGradient += thisDeltaPtr[outputDepthShift + i + j * outputWidth];

				}
			}


			// UPDATE WEIGHT -----------------------------

			// add regularization
			gradient += L1Lambda * sign(filterPtr[idx]) + L2Lambda * filterPtr[idx];
			gradient /= batchSize;

			// adadelta:
			adaSquares[idx] = ro * adaSquares[idx] + (1 - ro) * gradient * gradient;
			float dx = -sqrtf((adaDeltas[idx] + epsilon) / (adaSquares[idx] + epsilon)) * gradient;
			adaDeltas[idx] = ro * adaDeltas[idx] + (1 - ro) * dx * dx;
			filterPtr[idx] += dx;

			// -----------------------------------------


			// UPDATE BIAS ---------------------------
			if (idx % filterSize == 0)
			{
				// bias usually doesn't get regularised
				//biasGradient += L1Lambda * sign(biasPtr[idx / filterSize]) + L2Lambda * biasPtr[idx / filterSize];
				biasGradient /= batchSize;
				int biasIdx = idx / filterSize;

				adaBiasSquares[biasIdx] = ro * adaBiasSquares[biasIdx] + (1 - ro) * biasGradient * biasGradient;
				float dx = -sqrtf((adaBiasDeltas[biasIdx] + epsilon) / (adaBiasSquares[biasIdx] + epsilon)) * biasGradient;
				adaBiasDeltas[biasIdx] = ro * adaBiasDeltas[biasIdx] + (1 - ro) * dx * dx;
				biasPtr[biasIdx] += dx;
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
//Includes for IntelliSense 
#define _SIZE_T_DEFINED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math.h>

#include "../Observers/ColorHelpers.cu"

extern "C"
{

	__device__ float activationProbability(float x, float sigma)
	{
		return 1.0 / (1.0 + expf(-sigma * x));
	}

	__device__ float activateRandomly(float probability, float random)
	{
		return random < probability;
	}


	// RBM Kernels ////////////////////////////////////////////////////////////////


	__global__ void RBMInputForwardKernel(
										float *inputPtr,
										float *outputPtr,
										float *biasPtr,
										bool applyBias,
										int thisLayerSize
										)
	{
		// i: current neuron id
		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
				+ blockDim.x * blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if (i < thisLayerSize)
		{
			float result = inputPtr[i];
			if (applyBias)
				result += biasPtr[i];
			outputPtr[i] = inputPtr[i];
		}
	}

	__global__ void RBMInputForwardAndStoreKernel(
										float *inputPtr,
										float *outputPtr,
										float *biasPtr,
										float *storePtr,
										bool applyBias,
										int thisLayerSize
										)
	{
		// i: current neuron id
		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
				+ blockDim.x * blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if (i < thisLayerSize)
		{
			float result = inputPtr[i];
			if (applyBias)
				result += biasPtr[i];
			outputPtr[i] = result;
			storePtr[i] = result;
		}
	}




	__global__ void RBMSamplePositiveKernel(
										float *inputPtr,
										float *outputPtr,
										float *positivePtr,
										int thisLayerSize, // = outputPtr size
										int weightCount // = prevLayerSize * thisLayerSize
										)
	{

		int weightIndex = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if (weightIndex < weightCount)
		{
			// i: prev. layer neuron id
			int i = weightIndex / thisLayerSize;
			// j: current layer neuron id
			int j = weightIndex % thisLayerSize;

			positivePtr[weightIndex] = inputPtr[i] * outputPtr[j];
		}
	}

	__global__ void RBMRandomActivationKernel(
									float					*outputPtr,
									float					*randomPtr,
									int						size
									)
	{

		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if (i < size)
		{
			outputPtr[i] = activateRandomly(outputPtr[i], randomPtr[i]);
		}
	}


	__global__ void RBMForwardKernel(
									float					*inputPtr,
									float					*outputPtr,
									float					*weightPtr,
									float					*biasPtr,
									float					sigma,
									int						prevLayerSize,
									int						thisLayerSize,
									bool					useDropoutMask,
									bool					useDropout,
									float					dropoutRate,
									float					*dropoutMask
									)
	{
		// i: prev. layer neuron id
		// j: current layer neuron id
		int i;
		int j = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if (j < thisLayerSize)
		{
			// dropout this neuron
			if (useDropout && useDropoutMask && !dropoutMask[j])
			{
				outputPtr[j] = 0;
			}
			else
			{
				float sum = 0.0;
				int index = j;
				for (i = 0; i < prevLayerSize; i++) {
					sum += weightPtr[index] * inputPtr[i];
					index += thisLayerSize;
				}
				 // add bias
				sum += biasPtr[j];

				float result = activationProbability(sum, sigma);

				// only used for reconstruction forward calls
				if (useDropout && !useDropoutMask && dropoutRate < 1)
					result /= dropoutRate;


				// set output value
				outputPtr[j] = result;
			}
			

		}
	}

	// This is the same as Forward, only stores output to another memory block in addition...
	__global__ void RBMForwardAndStoreKernel(
									float					*inputPtr,
									float					*outputPtr,
									float					*weightPtr,
									float					*biasPtr,
									float					*storedOutputPtr,
									float					sigma,
									int						prevLayerSize,
									int						thisLayerSize,
									bool					useDropout,
									float					*dropoutMask
									)
	{
		// i: prev. layer neuron id
		// j: current layer neuron id
		int i;
		int j = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if (j < thisLayerSize)
		{
			// dropout this neuron
			if (useDropout && !dropoutMask[j])
			{
				outputPtr[j] = 0;
				storedOutputPtr[j] = 0;
			}
			else
			{
				float sum = 0.0;
				int index = j;
				for (i = 0; i < prevLayerSize; i++) {
					sum += weightPtr[index] * inputPtr[i];
					index += thisLayerSize;
				}
				 // add bias
				sum += biasPtr[j];

				float result = activationProbability(sum, sigma);

				// set output value
				outputPtr[j] = result;

				// store output value
				storedOutputPtr[j] = result;
			}
		}
	}

	__global__ void RBMBackwardKernel(
										float *inputPtr, // output of layer x+1 == this layer
										float *outputPtr, // output of layer x (we are going backwards) = input of layer x+1
										float *weightPtr,
										float *biasPtr, // biases of the layer x == previous layer
										float sigma,
										int prevLayerSize,
										int thisLayerSize
										)
	{
		// i: prev. layer neuron id
		// j: current layer neuron id
		int j;
		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if (i < prevLayerSize)
		{
			float sum = 0.0;

			int index = i * thisLayerSize;
			for (j = 0; j < thisLayerSize; j++)
				sum += weightPtr[index + j] * inputPtr[j];

			sum += biasPtr[i];

			// set output value
			outputPtr[i] = activationProbability(sum, sigma);

		}
	}

	__global__ void RBMUpdateBiasesKernel(
										float *biasPtr,
										float *positivePtr, // previous output of this layer == created by Forw&Store kernel
										float *negativePtr, // current output of this layer == outputPtr
										float *previousDeltaPtr,
										float *energyPtr,
										float learningRate,
										float momentum,
										float weightDecay,
										int thisLayerSize,
										bool storeEnergy
										)
	{

		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if (i < thisLayerSize)
		{
			float difference = positivePtr[i] - negativePtr[i];
			
			float delta =
				// gradient descent
				learningRate * difference +
				// momentum
				momentum * previousDeltaPtr[i] -
				// weight decay
				weightDecay * biasPtr[i] * learningRate;

			previousDeltaPtr[i] = delta;

			biasPtr[i] += delta;

			if (storeEnergy)
				atomicAdd(energyPtr, difference * difference);
		}
	}

	__global__ void RBMUpdateWeightsKernel(
										float *inputPtr,
										float *outputPtr,
										float *weightPtr,
										float *positivePtr,
										float *previousDeltaPtr,
										float *energyPtr,
										float learningRate,
										float momentum,
										float weightDecay,
										int thisLayerSize, // = outputPtr size
										int weightCount, // = prevLayerSize * thisLayerSize
										bool storeEnergy
										)
	{

		int weightIndex = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if (weightIndex < weightCount)
		{
			// first, compute negative (computed exactly as positive):
			// i: prev. layer neuron id
			int i = weightIndex / thisLayerSize;
			// j: current layer neuron id
			int j = weightIndex % thisLayerSize;

			float negative = inputPtr[i] * outputPtr[j];

			float difference = positivePtr[weightIndex] - negative;

			float delta =
				// gradient descent
				learningRate * (difference) +
				// momentum
				momentum * previousDeltaPtr[weightIndex] -
				// weight decay
				weightDecay * weightPtr[weightIndex] * learningRate;

			previousDeltaPtr[weightIndex] = delta;

			weightPtr[weightIndex] += delta;

			if (storeEnergy)
				atomicAdd(energyPtr, difference * difference);
		}
	}

	__global__ void RBMCopyFilterKernel(
		float *weightPtr,
		float *filterPtr,
		int weightCount,
		int i,
		int thisLayerSize
		)
	{

		int weightIndex = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if (weightIndex < weightCount)
		{
			filterPtr[weightIndex] = weightPtr[i + weightIndex * thisLayerSize];
		}
	}

	__global__ void RBMFilterObserver(float* refVectors, float* activations, int patchCount, int patchWidth, int patchHeight, 
		float minValue, float maxValue,
		int textureWidth, int textureHeight, unsigned int* pixels) 
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;	

		int size = textureWidth * textureHeight;
		int patchSize = patchWidth * patchHeight;

		int patchIndex = threadId % patchCount;
		int pixelIndex = threadId / patchCount;

		int pixX = pixelIndex % patchWidth;
		int pixY = pixelIndex / patchWidth;

		int patchesInRow = textureWidth / patchWidth;		
		
		int patchX = patchIndex % patchesInRow;
		int patchY = patchIndex / patchesInRow;

		if (threadId < size) 
		{			
			float activation = activations[patchIndex];
			float hue = (1 - activation) *  0.2f;
			float saturation = 0.8 * (activation > 0.05f);
			float value = scale_to_interval(refVectors[threadId], minValue, maxValue);
			pixels[(patchY * patchHeight + pixY) * textureWidth + patchX * patchWidth + pixX] = hsva_to_uint_rgba(hue, saturation, value, 1);
		}
	}

	__global__ void RBMDropoutMaskKernel(
		float *maskPtr,
		float dropout,
		int thisLayerSize
		)
	{

		int index = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if (index < thisLayerSize)
		{
			maskPtr[index] = dropout < maskPtr[index];
		}
	}

}
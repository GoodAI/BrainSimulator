#define _SIZE_T_DEFINED 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __cplusplus 
#define __cplusplus 
#endif

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>

#include "utils.cu"


extern "C"
{
	
    // Example of diposition of featureInfo
    // 
    //  ||    NbSources    ||      Offset     ||              SourceId             ||
    //  |-------------------------------------------------------------------------|
    //  ||  1  |  3  |  2  ||  0  |  1  |  4  ||  0  |  1  |  2  |  4  |  0  |  3  ||
    //  ----------------------------------------------------------------------------
    //      |     |    |       |     |     |      ^     ^                 ^
    //      |     |    |       ------|-----|------      |                 |
    //      |     |    |             ------|-------------                 |
    //      |     |    |                   --------------------------------
    //      |     |    | 
    //      |     |    |                        _____
    //       -----|----|---------------------------  _____ _____ _____
    //             ----|---------------------------------------       _____ _____
    //                  -----------------------------------------------------
    //



	__global__ void ForwardKernel(
									uint* featureInfo,
									MyLayerDim*	OutputDataPtr,
									MyLayerDim*	InputDataPtr,
									MyLayerDim*	WeightDataPtr,
									MyLayerDim*	BiasDataPtr,
									uint xStride,
									uint yStride
								)
	{
		int outputId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		__shared__ MyLayerDim s_input;
		__shared__ MyLayerDim s_output;
		__shared__ MyLayerDim s_weight;
		__shared__ MyLayerDim s_bias;

		if (threadIdx.x == 0)
		{
			s_input = *InputDataPtr;
			s_output = *OutputDataPtr;
			s_weight = *WeightDataPtr;
			s_bias = *BiasDataPtr;
		}

		__syncthreads();

		if (outputId < s_output.Count)
		{

			uint* nbSourcesArray = featureInfo;
			uint* sourceOffsetArray = featureInfo + s_output.Nb;
			uint* inputOffsetArray = sourceOffsetArray + s_output.Nb;

			int outputZ = outputId / s_output.Size;
			int outputY = (outputId - outputZ * s_output.Size) / s_output.Width;
			int outputX = outputId - outputZ * s_output.Size - outputY * s_output.Width;
		
			int inputX = xStride * outputX; // Top-left corner
			int inputY = yStride * outputY; // Top-left corner

			uint nbSources = nbSourcesArray[outputZ];
			uint sourceOffset = sourceOffsetArray[outputZ];
			float* weightPtr = s_weight.Ptr + sourceOffset * s_weight.Size;
			uint* inputIndexes = inputOffsetArray + sourceOffset;

			float sum = 0;
			for (int sourceNb = 0; sourceNb < nbSources; sourceNb++)
			{
				int inputOffsetZ = inputIndexes[sourceNb] * s_input.Size;
				int kernelOffsetZ = sourceNb * s_weight.Size;
				for (int y = 0; y < s_weight.Height; y++)
				{
					int inputLocalY = inputY + y;
					int inputOffsetYZ = inputLocalY * s_input.Width + inputOffsetZ;
					int kernelOffsetYZ = y * s_weight.Width + kernelOffsetZ;
					for (int x = 0; x < s_weight.Width; x++)
					{
						int inputLocalX = inputX + x;
						int inputOffsetXYZ = inputLocalX + inputOffsetYZ;
						int kernelOffsetXYZ = x + kernelOffsetYZ;

						float weightValue = weightPtr[kernelOffsetXYZ];
						float inputValue = s_input.Ptr[inputOffsetXYZ];

						sum += weightValue * inputValue;
					}
				}
			}

			s_output.Ptr[outputId] = sum + s_bias.Ptr[outputZ];
		}
	}

	




	__global__ void BackwardKernel(
									uint* featureInfo,
									MyLayerDim*	DeltaDataPtr,
									MyLayerDim*	PreviousLayerDeltaDataPtr,
									MyLayerDim*	WeightDataPtr,
									uint xStride,
									uint yStride
								)
	{
		int outputId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		__shared__ MyLayerDim s_weight;
		__shared__ MyLayerDim s_delta;
		__shared__ MyLayerDim s_previousLayerDelta;

		if (threadIdx.x == 0)
		{
			s_weight = *WeightDataPtr;
			s_delta = *DeltaDataPtr;
			s_previousLayerDelta = *PreviousLayerDeltaDataPtr;
		}

		__syncthreads();


		if (outputId < s_delta.Count)
		{

			uint* nbSourcesArray = featureInfo;
			uint* sourceOffsetArray = featureInfo + s_delta.Nb;
			uint* inputOffsetArray = sourceOffsetArray + s_delta.Nb;

			int outputZ = outputId / s_delta.Size;
			int outputY = (outputId - outputZ * s_delta.Size) / s_delta.Width;
			int outputX = outputId - outputZ * s_delta.Size - outputY * s_delta.Width;

			int deltaX = xStride * outputX; // Top-left corner
			int deltaY = yStride * outputY; // Top-left corner

			uint nbSources = nbSourcesArray[outputZ];
			uint sourceOffset = sourceOffsetArray[outputZ];
			float* weightPtr = s_weight.Ptr + sourceOffset * s_weight.Size;
			uint* inputIndexes = inputOffsetArray + sourceOffset;
			
			for (int sourceNb = 0; sourceNb < nbSources; sourceNb++)
			{
				int deltaOffsetZ = inputIndexes[sourceNb] * s_previousLayerDelta.Size;
				int kernelOffsetZ = sourceNb * s_weight.Size;
				for (int y = 0; y < s_weight.Height; y++)
				{
					int deltaLocalY = deltaY + y;
					int deltaOffsetYZ = deltaLocalY * s_previousLayerDelta.Width + deltaOffsetZ;
					int kernelOffsetYZ = y * s_weight.Width + kernelOffsetZ;
					for (int x = 0; x < s_weight.Width; x++)
					{
						int deltaLocalX = deltaX + x;
						int deltaOffsetXYZ = deltaLocalX + deltaOffsetYZ;
						int kernelOffsetXYZ = x + kernelOffsetYZ;

						float weightValue = weightPtr[kernelOffsetXYZ];
					
						float partialDelta = weightValue * s_delta.Ptr[outputId];
						float* address = s_previousLayerDelta.Ptr + deltaOffsetXYZ;

						atomicAdd(address, partialDelta);

					}
				}
			}
		}
	}






	__global__ void WeightKernel(
									uint* featureInfo,
									MyLayerDim*	InputDataPtr,
									MyLayerDim*	DeltaDataPtr,
									MyLayerDim*	WeightChangeDataPtr,
									MyLayerDim*	BiasChangeDataPtr,
									uint xStride,
									uint yStride
							)
	{
		int outputId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		
		__shared__ MyLayerDim s_input;
		__shared__ MyLayerDim s_delta;
		__shared__ MyLayerDim s_weightChange;
		__shared__ MyLayerDim s_biasChange;

		if (threadIdx.x == 0)
		{
			s_input = *InputDataPtr;
			s_delta = *DeltaDataPtr;
			s_weightChange = *WeightChangeDataPtr;
			s_biasChange = *BiasChangeDataPtr;
		}

		__syncthreads();


		if (outputId < s_delta.Count)
		{

			uint* nbSourcesArray = featureInfo;
			uint* sourceOffsetArray = featureInfo + s_delta.Nb;
			uint* inputOffsetArray = sourceOffsetArray + s_delta.Nb;

			int outputZ = outputId / s_delta.Size;
			int outputY = (outputId - outputZ * s_delta.Size) / s_delta.Width;
			int outputX = outputId - outputZ * s_delta.Size - outputY * s_delta.Width;
			
			int inputX = xStride * outputX; // Top-left corner
			int inputY = yStride * outputY; // Top-left corner

			uint nbSources = nbSourcesArray[outputZ];
			uint sourceOffset = sourceOffsetArray[outputZ];
			float* weightChangePtr = s_weightChange.Ptr + sourceOffset * s_weightChange.Size;
			uint* inputIndexes = inputOffsetArray + sourceOffset;

			float outputDelta = s_delta.Ptr[outputId];
			
			for (int sourceNb = 0; sourceNb < nbSources; sourceNb++)
			{
				int inputOffsetZ = sourceNb * s_input.Size;
				int kernelOffsetZ = inputIndexes[sourceNb] * s_weightChange.Size;
				for (int y = 0; y < s_weightChange.Height; y++)
				{
					int inputLocalY = inputY + y;
					int inputOffsetYZ = inputLocalY * s_input.Width + inputOffsetZ;
					int kernelOffsetYZ = y * s_weightChange.Width + kernelOffsetZ;
					for (int x = 0; x < s_weightChange.Width; x++)
					{
						int inputLocalX = inputX + x;
						int inputOffsetXYZ = inputLocalX + inputOffsetYZ;
						int kernelOffsetXYZ = x + kernelOffsetYZ;
					
						// dw = -learningRate * delta * SynapseOutput
						// We dont store the (- learningRate) part, this will be done once when we apply the batch

						float inputValue = s_input.Ptr[inputOffsetXYZ];

						float dw = outputDelta * inputValue;
						
						atomicAdd(weightChangePtr + kernelOffsetXYZ, dw);
					}
				}
			}
		
			// Bias change
			float dw = outputDelta * 1;
			atomicAdd(s_biasChange.Ptr + outputZ, dw); 
		}
	}



}

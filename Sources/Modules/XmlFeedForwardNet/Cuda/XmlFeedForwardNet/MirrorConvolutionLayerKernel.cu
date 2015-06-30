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
	__global__ void ForwardKernel(
									uint* featureInfo,
									MyLayerDim*	InputDataPtr,
									MyLayerDim*	OutputDataPtr,
									MyLayerDim*	WeightDataPtr,
									uint xStride,
									uint yStride
							)
	{
		int inputId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		__shared__ MyLayerDim s_input;
		__shared__ MyLayerDim s_output;
		__shared__ MyLayerDim s_weight;

		if (threadIdx.x == 0)
		{
			s_input = *InputDataPtr;
			s_output = *OutputDataPtr;
			s_weight = *WeightDataPtr;
		}

		__syncthreads();



		if (inputId < s_input.Count)
		{

			uint* nbSourcesArray = featureInfo;
			uint* sourceOffsetArray = featureInfo + s_input.Nb;
			uint* inputOffsetArray = sourceOffsetArray + s_input.Nb;

			int inputZ = inputId / s_input.Size;
			int inputY = (inputId - inputZ * s_input.Size) / s_input.Width;
			int inputX = inputId - inputZ * s_input.Size - inputY * s_input.Width;			
				
			int outputX = xStride * inputX; // Top-left corner
			int outputY = yStride * inputY; // Top-left corner
			

			uint nbSources = nbSourcesArray[inputZ];
			uint sourceOffset = sourceOffsetArray[inputZ];
			float* weightPtr = s_weight.Ptr + sourceOffset * s_weight.Size;
			uint* inputIndexes = inputOffsetArray + sourceOffset;

			for (int sourceNb = 0; sourceNb < nbSources; sourceNb++)
			{
				int outputOffsetZ = inputIndexes[sourceNb] * s_output.Size;
				int kernelOffsetZ = sourceNb * s_weight.Size;
				for (int y = 0; y < s_weight.Height; y++)
				{
					int outputLocalY = outputY + y;
					int outputOffsetYZ = outputLocalY * s_output.Width + outputOffsetZ;
					int kernelOffsetYZ = y * s_weight.Width + kernelOffsetZ;
					for (int x = 0; x < s_weight.Width; x++)
					{
						int outputLocalX = outputX + x;
						int outputOffsetXYZ = outputLocalX + outputOffsetYZ;
						int kernelOffsetXYZ = x + kernelOffsetYZ;

						float weightValue = weightPtr[kernelOffsetXYZ];

						float partialOutputValue = weightValue * s_input.Ptr[inputId];

						atomicAdd(s_output.Ptr + outputOffsetXYZ, partialOutputValue);
					}
				}
			}
		}
	}



	
	__global__ void ForwardBiasKernel(
									MyLayerDim*	OutputDataPtr,
									MyLayerDim*	BiasDataPtr
								)
	{
		int outputId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		
		__shared__ MyLayerDim s_output;
		__shared__ MyLayerDim s_bias;

		if (threadIdx.x == 0)
		{
			s_output = *OutputDataPtr;
			s_bias = *BiasDataPtr;
		}

		__syncthreads();

		if(outputId < s_output.Count)
		{
			s_output.Ptr[outputId] += s_bias.Ptr[outputId];
		}
	}



	
	__global__ void BackwardKernel(
									uint* featureInfo,
									MyLayerDim*	OutputDataPtr,
									MyLayerDim*	DeltaDataPtr,
									MyLayerDim*	PreviousLayerDeltaDataPtr,
									MyLayerDim*	WeightDataPtr,
									uint xStride,
									uint yStride
							)
	{
		int inputId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		
		__shared__ MyLayerDim s_output;
		__shared__ MyLayerDim s_delta;
		__shared__ MyLayerDim s_previousLayerDelta;
		__shared__ MyLayerDim s_weight;

		if (threadIdx.x == 0)
		{
			s_output = *OutputDataPtr;
			s_delta = *DeltaDataPtr;
			s_previousLayerDelta = *PreviousLayerDeltaDataPtr;
			s_weight = *WeightDataPtr;
		}

		__syncthreads();

		if (inputId < s_previousLayerDelta.Count)
		{

			uint* nbSourcesArray = featureInfo;
			uint* sourceOffsetArray = featureInfo + s_previousLayerDelta.Nb;
			uint* inputOffsetArray = sourceOffsetArray + s_previousLayerDelta.Nb;
			
			int inputZ = inputId / s_previousLayerDelta.Size;
			int inputY = (inputId - inputZ * s_previousLayerDelta.Size) / s_previousLayerDelta.Width;
			int inputX = inputId - inputZ * s_previousLayerDelta.Size - inputY * s_previousLayerDelta.Width;			
				
			int outputX = xStride * inputX; // Top-left corner
			int outputY = yStride * inputY; // Top-left corner
			

			uint nbSources = nbSourcesArray[inputZ];
			uint sourceOffset = sourceOffsetArray[inputZ];
			float* weightPtr = s_weight.Ptr + sourceOffset * s_weight.Size;
			uint* inputIndexes = inputOffsetArray + sourceOffset;
								
			// CALCULATE THE DELTA OF THE PREVIOUS LAYER

			float previousLayerDeltaSum = 0;		

			for (int sourceNb = 0; sourceNb < nbSources; sourceNb++)
			{
				int outputOffsetZ = inputIndexes[sourceNb] * s_delta.Size;
				int kernelOffsetZ = sourceNb * s_weight.Size;
				for (int y = 0; y < s_weight.Height; y++)
				{
					int outputLocalY = outputY + y;
					int outputOffsetYZ = outputLocalY * s_output.Width + outputOffsetZ;
					int kernelOffsetYZ = y * s_weight.Width + kernelOffsetZ;
					for (int x = 0; x < s_weight.Width; x++)
					{
						int outputLocalX = outputX + x;
						int outputOffsetXYZ = outputLocalX + outputOffsetYZ;
						int kernelOffsetXYZ = x + kernelOffsetYZ;

						float weightValue = weightPtr[kernelOffsetXYZ];

						float currentLayerDeltaValue = s_delta.Ptr[outputOffsetXYZ];

						previousLayerDeltaSum += weightValue * currentLayerDeltaValue;
					}
				}
			}
		
			s_previousLayerDelta.Ptr[inputId] = previousLayerDeltaSum;
		}
	}






	
	__global__ void WeightKernel(
									uint* featureInfo,
									MyLayerDim*	InputDataPtr,
									MyLayerDim*	DeltaDataPtr,
									MyLayerDim*	WeightChangeDataPtr,
									uint xStride,
									uint yStride
							)
	{
		int inputId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		__shared__ MyLayerDim s_input;
		__shared__ MyLayerDim s_delta;
		__shared__ MyLayerDim s_weightChange;

		if (threadIdx.x == 0)
		{
			s_input = *InputDataPtr;
			s_delta = *DeltaDataPtr;
			s_weightChange = *WeightChangeDataPtr;
		}

		__syncthreads();


		if (inputId < s_input.Count)
		{

			uint* nbSourcesArray = featureInfo;
			uint* sourceOffsetArray = featureInfo + s_input.Nb;
			uint* inputOffsetArray = sourceOffsetArray + s_input.Nb;
			
			int inputZ = inputId / s_input.Size;
			int inputY = (inputId - inputZ * s_input.Size) / s_input.Width;
			int inputX = inputId - inputZ * s_input.Size - inputY * s_input.Width;			
		
			int outputX = xStride * inputX; // Top-left corner
			int outputY = yStride * inputY; // Top-left corner
		

			uint nbSources = nbSourcesArray[inputZ];
			uint sourceOffset = sourceOffsetArray[inputZ];
			float* weightChangePtr = s_weightChange.Ptr + sourceOffset * s_weightChange.Size;
			uint* inputIndexes = inputOffsetArray + sourceOffset;


			for (int sourceNb = 0; sourceNb < nbSources; sourceNb++)
			{
				int outputOffsetZ = inputIndexes[sourceNb] * s_delta.Size;
				int kernelOffsetZ = sourceNb * s_weightChange.Size;
				for (int y = 0; y < s_weightChange.Height; y++)
				{
					int outputLocalY = outputY + y;
					int outputOffsetYZ = outputLocalY * s_delta.Width + outputOffsetZ;
					int kernelOffsetYZ = y * s_weightChange.Width + kernelOffsetZ;
					for (int x = 0; x < s_weightChange.Width; x++)
					{
						int outputLocalX = outputX + x;
						int outputOffsetXYZ = outputLocalX + outputOffsetYZ;
						int kernelOffsetXYZ = x + kernelOffsetYZ;

						float inputValue = s_input.Ptr[inputId];
						float deltaValue = s_delta.Ptr[outputOffsetXYZ];

						float dw = deltaValue * inputValue;

						atomicAdd(weightChangePtr + kernelOffsetXYZ, dw);
					}
				}
			}
		}
	}





	
	__global__ void WeightBiasKernel(
							MyLayerDim*	DeltaDataPtr,
							MyLayerDim*	BiasChangeDataPtr
						)
	{
		int weightId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		__shared__ MyLayerDim s_delta;
		__shared__ MyLayerDim s_biasChange;

		if (threadIdx.x == 0)
		{
			s_delta = *DeltaDataPtr;
			s_biasChange = *BiasChangeDataPtr;
		}

		__syncthreads();

		if (weightId < s_biasChange.Count)
		{
			s_biasChange.Ptr[weightId] += s_delta.Ptr[weightId];
		}
	}


}

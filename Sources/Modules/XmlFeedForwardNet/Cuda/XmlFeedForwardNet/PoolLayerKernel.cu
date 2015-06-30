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
									int				Stride,
									int				PoolRule,
									MyLayerDim*		OutputDataPtr,
									MyLayerDim*		InputDataPtr,
									int*			ChosenInput
								)
	{
		int outputId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		__shared__ MyLayerDim s_input;
		__shared__ MyLayerDim s_output;

		if (threadIdx.x == 0)
		{
			s_input = *InputDataPtr;
			s_output = *OutputDataPtr;
		}

		__syncthreads();

		if(outputId < s_output.Count)
		{
		
			int outputZ = outputId / s_output.Size;
			int outputY = (outputId - outputZ * s_output.Size) / s_output.Width;
			int outputX = outputId - outputZ * s_output.Size - outputY * s_output.Width;

			int inputZ = outputZ;
			int inputY = outputY * Stride;
			int inputX = outputX * Stride;
		
			float* inputWithOffset = s_input.Ptr + inputZ * s_input.Size;

			float outputValue;

			if (PoolRule == 0) // MAX POOLING
			{
				float maxValue = -9e9;
				int bestInputId = -1;
				for (int y = 0; y < Stride; y++)
				{
					int inputOffsetY = (inputY + y) * s_input.Width;
					for (int x = 0; x < Stride; x++)
					{
						int inputId = inputOffsetY + (inputX + x);
						float value = inputWithOffset[inputId];
						if (value > maxValue)
						{
							maxValue = value;
							bestInputId = inputId;
						}
					}
				}

				ChosenInput[outputId] = inputZ * s_input.Size + bestInputId;

				outputValue = maxValue;
			}
			else if (PoolRule == 1) // AVERAGE POOLING
			{
				float sum = 0;
				for (int y = 0; y < Stride; y++)
				{
					int inputOffsetY = (inputY + y) * s_input.Width;
					for (int x = 0; x < Stride; x++)
					{
						int inputId = inputOffsetY + (inputX + x);
						float value = inputWithOffset[inputId];
						sum += value;
					}
				}
				outputValue = sum / (Stride * Stride);
			}
			else
			{
				// Add new rule
				outputValue = -1;
			}
		
			s_output.Ptr[outputId] = outputValue;
		}
	}





	
	__global__ void BackwardKernel(
								int				PoolRule,
								int				Stride,
								MyLayerDim*		DeltaDataPtr,
								MyLayerDim*		PreviousLayerDeltaDataPtr,
								int*			ChosenInput
							)
	{
		int inputId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		__shared__ MyLayerDim s_delta;
		__shared__ MyLayerDim s_previousLayerDelta;

		if (threadIdx.x == 0)
		{
			s_delta = *DeltaDataPtr;
			s_previousLayerDelta = *PreviousLayerDeltaDataPtr;
		}

		__syncthreads();


		if(inputId < s_previousLayerDelta.Count)
		{
			int inputZ = inputId / s_previousLayerDelta.Size;
			int inputY = (inputId - inputZ * s_previousLayerDelta.Size) / s_previousLayerDelta.Width;
			int inputX = inputId - inputZ * s_previousLayerDelta.Size - inputY * s_previousLayerDelta.Width;

			int outputZ = inputZ;
			int outputY = inputY / Stride;
			int outputX = inputX / Stride;

			float previousLayerDeltaValue = -1;

			// Find out if this inputId has been used during the last forward propagation


			float chosenInputId = ChosenInput[outputZ * s_delta.Size + outputY * s_delta.Width + outputX];

			if (PoolRule == 0) // MAX POOLING
			{
				if (inputId == chosenInputId)
				{
					// This inputId has been used for the forward prop
					float dE_by_dnet = s_delta.Ptr[outputZ * s_delta.Size + outputY * s_delta.Width + outputX];

					// The activation function is f(x) = x.
					// The derivative is f'(x) = 1.
					float do_by_dnet = 1;
		
					previousLayerDeltaValue = dE_by_dnet * do_by_dnet;

				}
				else
				{
					// This inputId has NOT been used.
					previousLayerDeltaValue = 0;
				}
			}
			else if (PoolRule == 1) // AVERAGE POOLING
			{
				float dE_by_dnet = s_delta.Ptr[outputZ * s_delta.Size + outputY * s_delta.Width + outputX];

				// The activation function is f(x) = x.
				// The derivative is f'(x) = 1.
				float do_by_dnet = 1;
		
				previousLayerDeltaValue = dE_by_dnet * do_by_dnet;
			}

			s_previousLayerDelta.Ptr[inputId] = previousLayerDeltaValue;
		}
	}
}

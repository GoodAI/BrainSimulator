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
									uint		PoolRule,
									uint		Stride,
									MyLayerDim*	OutputDataPtr,
									MyLayerDim*	InputDataPtr
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
			int inputY = outputY / Stride;
			int inputX = outputX / Stride;
		
			int inputId = inputZ * s_input.Size + inputY * s_input.Width + inputX;
		
			float outputValue = -1;

			if (PoolRule == 0) // MAX POOLING
			{
				// FIXME
				outputValue = -1;
			}
			else if (PoolRule == 1) // AVERAGE POOLING
			{
				outputValue = s_input.Ptr[inputId];
			}
		
			s_output.Ptr[outputId] = outputValue;
		}
	}







	
	__global__ void BackwardKernel(
								uint		PoolRule,
								uint		Stride,
								MyLayerDim*	DeltaDataPtr,
								MyLayerDim*	PreviousLayerDeltaDataPtr
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
			int outputY = inputY * Stride;
			int outputX = inputX * Stride;

			int outputOffsetZ = outputZ * s_delta.Size;
		
			float previousLayerDeltaValue;

			if (PoolRule == 0) // MAX POOLING
			{
				//// FIXME
				previousLayerDeltaValue = -1;
			}
			else if (PoolRule == 1) // AVERAGE POOLING
			{
				float sum = 0;
				for (int y = 0; y < Stride; y++)
				{
					int outputOffsetY = outputOffsetZ + (outputY + y) * s_delta.Width;
					for (int x = 0; x < Stride; x++)
					{
						int outputId = outputOffsetY + (outputX + x);

						float value = s_delta.Ptr[outputId];
						sum += value;
					}
				}
				previousLayerDeltaValue = sum / (Stride * Stride);
			}

			s_previousLayerDelta.Ptr[inputId] = previousLayerDeltaValue;
		}
	}

}

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
		
				
			// Since we are using double, we cannot use exponential for numbers above 709.
			// We then have to use a nice property of the softmax function.
			// I can remove any value to all the values, and the softmax result will be equal
			// In this case, we will remove the maximum value.

			// Find the maximmum value
			double maxValue = -9e9;
			for (int x = 0; x < s_input.Count; x++)
			{
				double inputValue = s_input.Ptr[x];
				if (inputValue > maxValue)
					maxValue = inputValue;
			}

			// Compute the sum
			double sum = 0;
			for (int x = 0; x < s_input.Count; x++)
			{
				double inputValue = s_input.Ptr[x] - maxValue;
				sum += exp(inputValue);
			}

			float outputValue = exp(s_input.Ptr[outputId] - maxValue) / sum;

			s_output.Ptr[outputId] = outputValue;

		}
	}




	
	__global__ void BackwardKernel(
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
			s_previousLayerDelta.Ptr[inputId] = s_delta.Ptr[inputId];
		}
	}
}

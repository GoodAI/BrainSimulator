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
	typedef enum MyEnumActivationFunction
	{
		NO_ACTIVATION = 0,
		LOGISTIC = 1,
		RELU = 2,
		TANH = 3
	} MyActivationFunction;


	__global__ void ForwardKernel(
							int			ActivationFunction,
							MyLayerDim*	OutputDataPtr,
							MyLayerDim*	InputDataPtr
						)
	{
		int neuronId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
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


		if(neuronId < s_output.Count)
		{
			if (ActivationFunction == LOGISTIC)
			{
				s_output.Ptr[neuronId] = (double)(1) / (   (double)(1) + (double)exp((double)(-s_input.Ptr[neuronId]))   );
			}
			else if (ActivationFunction == RELU)
			{
				float input = s_input.Ptr[neuronId];
				s_output.Ptr[neuronId] = (input > 0) ? input : 0;
			}
			else if (ActivationFunction == TANH)
			{
				s_output.Ptr[neuronId] = tanh(s_input.Ptr[neuronId]);
			}
			else // if ActivationFunction == 0 or unknown : no activation function
			{
				s_output.Ptr[neuronId] = s_input.Ptr[neuronId];
			}
		}
	}



	
	__global__ void BackwardKernel(
							int			ActivationFunction,
							MyLayerDim*	OutputDataPtr,
							MyLayerDim*	DeltaDataPtr,
							MyLayerDim*	PreviousLayerDeltaDataPtr
								)
	{
		int inputId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		__shared__ MyLayerDim s_delta;
		__shared__ MyLayerDim s_previousLayerDelta;
		__shared__ MyLayerDim s_output;

		if (threadIdx.x == 0)
		{
			s_delta = *DeltaDataPtr;
			s_previousLayerDelta = *PreviousLayerDeltaDataPtr;
			s_output = *OutputDataPtr;
		}

		__syncthreads();


		if(inputId < s_previousLayerDelta.Count)
		{
			// CALCULATE THE DELTA OF THE PREVIOUS LAYER
			if (ActivationFunction == LOGISTIC)
			{
				// log(input) = output, so we can reuse output directly
				float logisticInput = s_output.Ptr[inputId];
				s_previousLayerDelta.Ptr[inputId] = (logisticInput * (1 - logisticInput)) * s_delta.Ptr[inputId];
			}
			else if (ActivationFunction == RELU)
			{
				s_previousLayerDelta.Ptr[inputId] = (s_output.Ptr[inputId] > 0 ? s_delta.Ptr[inputId] : 0);
			}
			else if (ActivationFunction == TANH)
			{
				// tanh(input) = output, so we can reuse output directly
				float tanhInput = s_output.Ptr[inputId];
				s_previousLayerDelta.Ptr[inputId] = (1 - tanhInput * tanhInput) * s_delta.Ptr[inputId];
			}
			else // if ActivationFunction == 0 or unknown
			{
				// No activation function.
				s_previousLayerDelta.Ptr[inputId] = s_delta.Ptr[inputId];
			}
		}
	}


}

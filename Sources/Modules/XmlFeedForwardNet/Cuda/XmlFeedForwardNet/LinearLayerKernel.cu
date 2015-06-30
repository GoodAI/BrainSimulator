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
							MyLayerDim*	InputDataPtr,
							MyLayerDim*	WeightDataPtr,
							MyLayerDim*	BiasDataPtr
						)
	{
		int neuronId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
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


		if(neuronId < s_output.Count)
		{				
			s_output.Ptr[neuronId] = s_input.Ptr[neuronId] * s_weight.Ptr[neuronId];// + s_bias.Ptr[neuronId];
		}
	}

	
	
	__global__ void BackwardKernel(
							MyLayerDim*	DeltaDataPtr,
							MyLayerDim*	WeightDataPtr,
							MyLayerDim*	PreviousLayerDeltaDataPtr
								)
	{
		int inputId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		__shared__ MyLayerDim s_delta;
		__shared__ MyLayerDim s_previousLayerDelta;
		__shared__ MyLayerDim s_weight;

		if (threadIdx.x == 0)
		{
			s_delta = *DeltaDataPtr;
			s_previousLayerDelta = *PreviousLayerDeltaDataPtr;
			s_weight = *WeightDataPtr;
		}

		__syncthreads();

		if(inputId < s_previousLayerDelta.Count)
		{
			// CALCULATE THE DELTA OF THE PREVIOUS LAYER
			s_previousLayerDelta.Ptr[inputId] = s_delta.Ptr[inputId] * s_weight.Ptr[inputId];
		}
	}




	
	__global__ void WeightKernel(
							MyLayerDim*	InputDataPtr,
							MyLayerDim*	DeltaDataPtr,
							MyLayerDim*	WeightChangeDataPtr
						)
	{
		int weightId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
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

		if(weightId < s_weightChange.Count)
		{
			// dw = -learningRate * delta * SynapseOutput
			// We dont store the (- learningRate) part, this will be done once when we apply the batch
		
			float deltaValue = s_delta.Ptr[weightId];
			float inputValue = s_input.Ptr[weightId]; // Synapse weightValue
			float dw = deltaValue * inputValue;

			s_weightChange.Ptr[weightId] += dw;
		}
	}


	
	
	__global__ void BiasKernel(
							MyLayerDim*	DeltaDataPtr,
							MyLayerDim*	BiasChangeDataPtr
						)
	{
		int biasId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
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

		if(biasId < s_biasChange.Count)
		{
			
			float inputValue = 1; // Bias weightValue

			float deltaValue = s_delta.Ptr[biasId];

			float dw = deltaValue * inputValue;

			s_biasChange.Ptr[biasId] += dw;
		}
	}
}

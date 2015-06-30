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
			float* weightBuffer = s_weight.Ptr + neuronId * s_input.Count;
				
			float sum = 0;
		
			for (int x = 0; x < s_input.Count; x++)
				sum += weightBuffer[x] * s_input.Ptr[x];

			s_output.Ptr[neuronId] = sum + s_bias.Ptr[neuronId]; // Bias
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

			float previousLayerDeltaSum = 0;		
			for (int i = 0; i < s_delta.Count; i++)
			{
				float currentLayerDelta = s_delta.Ptr[i];
				float weightValue = s_weight.Ptr[inputId + i * s_previousLayerDelta.Count];
				previousLayerDeltaSum += currentLayerDelta * weightValue;
			}
			s_previousLayerDelta.Ptr[inputId] = previousLayerDeltaSum;
		}
	}




	// only calculate weight changes
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
		
			float deltaValue = s_delta.Ptr[weightId / s_input.Count];

			int inputNeuronId = weightId % s_input.Count;
			float inputValue = s_input.Ptr[inputNeuronId]; // Synapse weightValue
			float dw = deltaValue * inputValue;

			s_weightChange.Ptr[weightId] += dw;
		}
	}


	// calculate weight changes AND apply them = update weights at the same time
	__global__ void BackpropKernel(
							MyLayerDim*	InputDataPtr,
							MyLayerDim*	DeltaDataPtr,
							MyLayerDim*	WeightDataPtr,
							MyLayerDim*	LastWeightDeltaPtr,
							float	learningRate,
							float   momentum,
							int count
						)
	{
		int weightId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
						+ blockDim.x * blockIdx.x				//blocks preceeding current block
						+ threadIdx.x;

		if(weightId < count)
		{
			// dw = -learningRate * delta * SynapseOutput
		
			float deltaValue = (*DeltaDataPtr).Ptr[weightId / (*InputDataPtr).Count];

			int inputNeuronId = weightId % (*InputDataPtr).Count;
			float inputValue = (*InputDataPtr).Ptr[inputNeuronId]; // Synapse weightValue

			float dw = deltaValue * inputValue;

			// update weights
			float weightDelta =  -learningRate * dw + momentum * (*LastWeightDeltaPtr).Ptr[weightId];
			(*WeightDataPtr).Ptr[weightId] += weightDelta;

			(*LastWeightDeltaPtr).Ptr[weightId] = weightDelta;
		}

	}


	
	// TODO – biases can be merged with weights
	__global__ void BiasKernel(
							MyLayerDim*	DeltaDataPtr,
							MyLayerDim*	BiasChangeDataPtr
						)
	{
		int biasId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
						+ blockDim.x * blockIdx.x				//blocks preceeding current block
						+ threadIdx.x;
		
		if(biasId < (*BiasChangeDataPtr).Count)
		{
			(*BiasChangeDataPtr).Ptr[biasId] += (*DeltaDataPtr).Ptr[biasId];
		}
	}
}

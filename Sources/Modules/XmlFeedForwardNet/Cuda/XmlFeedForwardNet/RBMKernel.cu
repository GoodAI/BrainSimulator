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
							MyLayerDim*	InputDataPtr,
							MyLayerDim*	OutputDataPtr,
							MyLayerDim*	WeightDataPtr,
							MyLayerDim*	BiasDataPtr,
							float* random
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

			sum += s_bias.Ptr[neuronId];

			float activationProb = 1.0 / (1 + exp(-sum));
			/*
			if (random[neuronId] < activationProb)
				s_output.Ptr[neuronId] = 1;
			else
				s_output.Ptr[neuronId] = 0;*/

			s_output.Ptr[neuronId] = activationProb;
			//TODO
			// assigning activationProb here and there and everywhere works...
			//s_output.Ptr[neuronId] = random[neuronId] < (1.0 / (1 + exp(-sum)));

		}
	}


	__global__ void ForwardAndStoreKernel(
							MyLayerDim*	InputDataPtr,
							MyLayerDim*	OutputDataPtr,
							MyLayerDim*	WeightDataPtr,
							MyLayerDim*	BiasDataPtr,
							MyLayerDim* StoreDataPtr,
							float* random
						)
	{
		int neuronId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		__shared__ MyLayerDim s_input;
		__shared__ MyLayerDim s_output;
		__shared__ MyLayerDim s_weight;
		__shared__ MyLayerDim s_bias;
		__shared__ MyLayerDim s_store;

		if (threadIdx.x == 0)
		{
			s_input = *InputDataPtr;
			s_output = *OutputDataPtr;
			s_weight = *WeightDataPtr;
			s_bias = *BiasDataPtr;
			s_store = *StoreDataPtr;
		}

		__syncthreads();


		if(neuronId < s_output.Count)
		{
			float* weightBuffer = s_weight.Ptr + neuronId * s_input.Count;
				
			float sum = 0;
		
			for (int x = 0; x < s_input.Count; x++)
				sum += weightBuffer[x] * s_input.Ptr[x];

			sum += s_bias.Ptr[neuronId];

			float activationProb = 1.0 / (1 + exp(-sum));
			/*
			if (random[neuronId] < activationProb)
				s_output.Ptr[neuronId] = 1;
			else
				s_output.Ptr[neuronId] = 0;

			s_store.Ptr[neuronId] = s_output.Ptr[neuronId];*/
			s_output.Ptr[neuronId] = activationProb;
			s_store.Ptr[neuronId] = activationProb;


		}
	}

	__global__ void BackwardKernel(
							MyLayerDim*	InputDataPtr,
							MyLayerDim*	OutputDataPtr,
							MyLayerDim*	WeightDataPtr,
							MyLayerDim*	BiasDataPtr,
							bool useBias,
							float* random
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

			float* weightBuffer = s_weight.Ptr + neuronId;

			int idx = 0;
				
			float sum = 0;
		
			for (int x = 0; x < s_input.Count; x++) {
				sum += weightBuffer[idx] * s_input.Ptr[x];
				idx += s_output.Count;
				// this is analogous to (but faster then)
				// sum += weightBuffer[x * s_output.Count] * s_input.Ptr[x]
			}

			if (useBias)
				//TODO fix
				sum += s_bias.Ptr[neuronId];

			float activationProb = 1.0 / (1 + exp(-sum));
			/*
			if (random[neuronId] < activationProb)
				s_output.Ptr[neuronId] = 1;
			else
				s_output.Ptr[neuronId] = 0;*/
			s_output.Ptr[neuronId] = activationProb;
		}
	}



	// samples weights (Positive or Negative) from hidden and visible layer activations
	// by default, only used for Positive, because UpdateWeightKernel computes Negative
	// but doesn't need to store it -> it is immediately used to compute new weights
	__global__ void SampleKernel(
							MyLayerDim*	InputDataPtr,
							MyLayerDim*	OutputDataPtr,
							float*	PositiveDataPtr,
							int weightCount
						)
	{
		int weightId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
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


		if(weightId < weightCount)
		{
			int inIdx = weightId % s_input.Count;
			int outIdx = weightId / s_input.Count;

			// this samples both positive and negative for      weight += positive - negative
			PositiveDataPtr[weightId] = s_input.Ptr[inIdx] * s_output.Ptr[outIdx];
			float f = inIdx + outIdx;
		}
	}


	// samples NEGATIVE and immediately uses it and previously calculated Positive to compute new weights
	__global__ void UpdateWeightKernel(
							float*	PositiveDataPtr,
							MyLayerDim*	InputDataPtr,
							MyLayerDim*	OutputDataPtr,
							MyLayerDim*	WeightDataPtr,
							float* lastDelta,
							float LearningRate,
							float Momentum,
							float WeightDecay,
							float* energy,
							bool saveEnergy
						)
	{
		int weightId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		__shared__ MyLayerDim s_weight;
		__shared__ MyLayerDim s_input;
		__shared__ MyLayerDim s_output;

		if (threadIdx.x == 0)
		{
			s_input = *InputDataPtr;
			s_output = *OutputDataPtr;
			s_weight = *WeightDataPtr;
		}

		__syncthreads();


		if(weightId < s_weight.Count)
		{
			int inIdx = weightId % s_input.Count;
			int outIdx = weightId / s_input.Count;

			// s_in[inIdx] * s_out[outIdx] ~== negative! ->      weight += positive - negative
			float negative = s_input.Ptr[inIdx] * s_output.Ptr[outIdx];

			float delta = PositiveDataPtr[weightId] - negative;
			if (saveEnergy)
				atomicAdd(energy, delta * delta);

			float oldWeight = s_weight.Ptr[weightId];

			float penalisation = WeightDecay * oldWeight * LearningRate;

			s_weight.Ptr[weightId] += LearningRate * delta + Momentum * lastDelta[weightId] - penalisation;
			lastDelta[weightId] = delta;
		}
	}


	// Update biases of one layer based on the NEURON's
	// activity at the start (Positive), activity at the end (Negative)
	// this is not the same Positive  array as above (that was for weights)
	__global__ void UpdateBiasKernel(
							MyLayerDim*	PositiveDataPtr,
							MyLayerDim*	NegativeDataPtr,
							MyLayerDim*	BiasDataPtr,
							float* lastDelta,
							float* energy,
							float LearningRate,
							float Momentum,
							float WeightDecay,
							bool saveEnergy
						)
	{
		int biasId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		__shared__ MyLayerDim s_positive;
		__shared__ MyLayerDim s_negative;
		__shared__ MyLayerDim s_bias;

		if (threadIdx.x == 0)
		{
			s_positive = *PositiveDataPtr;
			s_negative = *NegativeDataPtr;
			s_bias = *BiasDataPtr;
			if (saveEnergy)
				energy[0] = 0;
		}

		__syncthreads();


		if(biasId < s_bias.Count)
		{
			float delta = s_positive.Ptr[biasId] - s_negative.Ptr[biasId];

			if (saveEnergy) 
				atomicAdd(energy, delta * delta);
			
			
			float oldBias = s_bias.Ptr[biasId];

			float penalisation = WeightDecay * oldBias * LearningRate;

			s_bias.Ptr[biasId] += LearningRate * (delta) + Momentum * lastDelta[biasId] - penalisation;
			lastDelta[biasId] = delta;
		}
	}



	__global__ void ObserverKernel(
							MyLayerDim*	DataPtr,
							float* target
						)
	{

		int id = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(id < (*DataPtr).Count)
		{
			target[id] = (*DataPtr).Ptr[id];
		}
	}

	__global__ void NeuronCopyForwardKernel(MyLayerDim* input, MyLayerDim* output, MyLayerDim* bias, float* random)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		if (id < (*output).Count) {

			float sum = (*input).Ptr[id] + (*bias).Ptr[id];

			float activationProb = 1.0 / (1 + exp(-sum));
			/*
			if (random[id] < activationProb)
				(*output).Ptr[id] = 1;
			else
				(*output).Ptr[id] = 0;*/

			(*output).Ptr[id] = activationProb;

		}
	}

	__global__ void NeuronCopyForwardAndStoreKernel(MyLayerDim* input, MyLayerDim* output, MyLayerDim* bias, MyLayerDim* stored, float* random)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		if (id < (*output).Count) {
			float sum = (*input).Ptr[id] + (*bias).Ptr[id];

			float activationProb = 1.0 / (1 + exp(-sum));
			/*
			if (random[id] < activationProb)
				(*output).Ptr[id] = 1;
			else
				(*output).Ptr[id] = 0;

			(*stored).Ptr[id] = (*output).Ptr[id];*/


			(*output).Ptr[id] = activationProb;
			(*stored).Ptr[id] = activationProb;
		}
	}

}
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
	// TODO - when not using batches (= when using online learning), updating weights should take place in 
	// NeuronLayerKernel->BackpropKernel and this kernel will not be needed (different layer types need their own kernel changes)
	__global__ void UpdateWeightKernel(
							uint	weightCount,
							float	learningRate,
							float   momentum,
							float*	weight,
							float*	weightChange,
							float*	lastWeightDelta
						)
	{
		int weightId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		if(weightId < weightCount)
		{
			// update weightValue
			float weightDelta = momentum * lastWeightDelta[weightId] - learningRate * weightChange[weightId];
			weight[weightId] += weightDelta;

			lastWeightDelta[weightId] = weightDelta;
			weightChange[weightId] = 0;
		}
	}
}

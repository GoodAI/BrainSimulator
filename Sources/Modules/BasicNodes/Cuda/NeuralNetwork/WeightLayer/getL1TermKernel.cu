//Includes for IntelliSense 
#define _SIZE_T_DEFINED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math.h>

extern "C"
{
	__global__ void getL1TermKernel(
		float *weightPtr,
		int weights
		)
	{
		//// i: prev. layer neuron id
		//// j: current layer neuron id
		//int i;
		//int j = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
		//	+ blockDim.x * blockIdx.x				//blocks preceeding current block
		//	+ threadIdx.x;

		//if (j < thisLayerSize)
		//{
		//	float sum = 0.0;
		//	int index = j;
		//	for (i = 0; i < prevLayerSize; i++) {
		//		sum += weightPtr[index] * inputPtr[i];
		//		index += thisLayerSize;
		//	}
		//	// add bias
		//	sum += biasPtr[j];

		//	// sum neuron input
		//	weightedInputPtr[j] = sum;

		//	// set output value
		//	outputPtr[j] = Evaluate(activationFunction, sum);
		//}
	}
}
//Includes for IntelliSense 
#define _SIZE_T_DEFINED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math.h>

#include "../NeuralNetwork/Activation/ActivationFunction.cu"


extern "C"  
{
	__constant__ int D_INPUT_UNITS;
	__constant__ int D_HIDDEN_UNITS;
	__constant__ int D_OUTPUT_UNITS;
	__constant__ ActivationFunctionEnum D_ACTIVATION_FUNCTION;
	

	__global__ void FeedforwardHiddenKernel(float *input, float *hiddenActivations, float *previousHiddenActivations, float *hiddenActivationDerivatives, float *inputWeights, float *recurrentWeights)
	{
		int unitId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (unitId < D_HIDDEN_UNITS)
		{
			float weightedSum = 0;

			int weightId = unitId * D_INPUT_UNITS;
			for (int i = 0; i < D_INPUT_UNITS; i++)
			{
				weightedSum += inputWeights[weightId] * input[i];
				weightId++;
			}

			weightId = unitId * D_HIDDEN_UNITS;
			for (int i = 0; i < D_HIDDEN_UNITS; i++)
			{
				weightedSum += recurrentWeights[weightId] * previousHiddenActivations[i];
				weightId++;
			}

			hiddenActivations[unitId] = Evaluate(D_ACTIVATION_FUNCTION, weightedSum);
			hiddenActivationDerivatives[unitId] = EvaluateDerivative(D_ACTIVATION_FUNCTION, weightedSum);
		}
	}

	__global__ void FeedforwardOutputKernel(float *hiddenActivations, float *outputActivations, float *outputActivationDerivatives, float *outputWeights)
	{
		int unitId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (unitId < D_OUTPUT_UNITS)
		{
			float weightedSum = 0;

			int weightId = unitId * D_HIDDEN_UNITS;
			for (int i = 0; i < D_HIDDEN_UNITS; i++)
			{
				weightedSum += outputWeights[weightId] * hiddenActivations[i];
				weightId++;
			}

			outputActivations[unitId] = Evaluate(D_ACTIVATION_FUNCTION, weightedSum);
			outputActivationDerivatives[unitId] = EvaluateDerivative(D_ACTIVATION_FUNCTION, weightedSum);
		}
	}
}

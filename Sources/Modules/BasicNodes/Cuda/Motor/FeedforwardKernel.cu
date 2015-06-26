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
	__constant__ int D_HIDDEN_UNIT_WEIGHTS;
	__constant__ ActivationFunctionEnum D_ACTIVATION_FUNCTION;
	
	//kernel code
	__global__ void FeedforwardKernel(float *activation, float *previousActivation, float *activationDerivative, float *weights)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (id < D_HIDDEN_UNITS + D_OUTPUT_UNITS)
		{
			bool isHiddenUnit = id < D_HIDDEN_UNITS;

			//rows to be skipped in the weights matrix
			int offset = isHiddenUnit * (id * (1 + D_INPUT_UNITS + D_HIDDEN_UNITS)); //hidden units are connected to threshold, input and hidden units
			offset += !isHiddenUnit * (D_HIDDEN_UNIT_WEIGHTS //if output unit then skip all hidden unit weights
					+ (id - D_HIDDEN_UNITS) * (1 + D_HIDDEN_UNITS)); //output units are connected to threshold and hidden units

			//compute weighted sum of input to the unit
			float weightedSum = weights[offset]; //threshold

			//loop through input units and if this is a hidden unit, sum their weighted activations
			for (int i = 0; i < D_INPUT_UNITS; i++)
			{
				weightedSum += isHiddenUnit * activation[1 + i] * weights[isHiddenUnit * (offset + 1 + i)];
			}
			//loop through hidden units and sum their weighted activation in previous time step
			for (int i = 0; i < D_HIDDEN_UNITS; i++)
			{
				weightedSum += previousActivation[1 + D_INPUT_UNITS + i] * weights[offset + 1 + isHiddenUnit * D_INPUT_UNITS + i];
			}

			activation[1 + D_INPUT_UNITS + id] = Evaluate(D_ACTIVATION_FUNCTION, weightedSum);
			activationDerivative[1 + D_INPUT_UNITS + id] = EvaluateDerivative(D_ACTIVATION_FUNCTION, weightedSum);
		}
	}
}

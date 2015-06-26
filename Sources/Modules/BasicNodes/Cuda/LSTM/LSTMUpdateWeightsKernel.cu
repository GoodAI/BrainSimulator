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
	__global__ void LSTMUpdateGateWeightsKernel(
		float *input,
		float *previousOutput,
		float *cellStates,
		float *cellStateErrors,
		float *outputGateDeltas,
		float *inputGateWeights,
		float *forgetGateWeights,
		float *outputGateWeights,
		float *inputGateWeightsRTRLPartials,
		float *forgetGateWeightsRTRLPartials,

		float learningRate,
		int inputCount,
		int previousOutputCount,
		int cellsPerBlock
		)
	{
		int weightId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		int weightsPerGate = inputCount + previousOutputCount + cellsPerBlock + 1;

		if (weightId < weightsPerGate * previousOutputCount/cellsPerBlock)
		{
			int fromId = weightId % weightsPerGate;
			int toId = weightId / weightsPerGate;

			//update output gate weight
			int isFromInputUnit = fromId >= 0 && fromId < inputCount;
			int isFromPreviousOutputUnit = (fromId >= inputCount) && (fromId < inputCount + previousOutputCount);
			int isPeephole = (fromId >= inputCount + previousOutputCount) && (fromId < inputCount + previousOutputCount + cellsPerBlock);
			int isFromBiasUnit = fromId == (inputCount + previousOutputCount + cellsPerBlock);

			float inputFromWeight = isFromInputUnit * input[isFromInputUnit * fromId]
									+ isFromPreviousOutputUnit * previousOutput[isFromPreviousOutputUnit * (fromId - inputCount)]
									+ isPeephole * cellStates[isPeephole * (toId * cellsPerBlock + (fromId - inputCount - previousOutputCount))]
									+ isFromBiasUnit * 1;
	
			outputGateWeights[weightId] += learningRate * outputGateDeltas[toId] * inputFromWeight;

			//update input and forget gate weights
			float inputGateWeightDelta = 0;
			float forgetGateWeightDelta = 0;
			//loop through cells
			for (int cellId = toId * cellsPerBlock; cellId < (toId + 1) * cellsPerBlock; cellId++)
			{
				inputGateWeightDelta += cellStateErrors[cellId] * inputGateWeightsRTRLPartials[cellId * weightsPerGate + fromId];
				forgetGateWeightDelta += cellStateErrors[cellId] * forgetGateWeightsRTRLPartials[cellId * weightsPerGate + fromId];
			}

			inputGateWeights[weightId] += inputGateWeightDelta;
			forgetGateWeights[weightId] += forgetGateWeightDelta;
		}
	}

	__global__ void LSTMUpdateCellWeightsKernel(
		float *input,
		float *previousOutput,
		float *cellStateErrors,
		float *cellInputWeights,
		float *cellWeightsRTRLPartials,

		float learningRate,
		int inputCount,
		int previousOutputCount,
		int cellsPerBlock
		)
	{
		int weightId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		int weightsPerCell = inputCount + previousOutputCount + 1;
		
		if (weightId < weightsPerCell * previousOutputCount)
		{
			int cellId = weightId / weightsPerCell;

			cellInputWeights[weightId] += learningRate * cellStateErrors[cellId] * cellWeightsRTRLPartials[weightId];
		}
	}
}

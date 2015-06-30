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
	__global__ void LSTMCellWeightsRTRLPartialsKernel(
		float *input,
		float *previousOutput,
		float *inputGateActivations,
		float *forgetGateActivations,
		float *cellInputActivationDerivatives,
		float *cellWeightsRTRLPartials,

		int inputCount,
		int previousOutputCount,
		int cellsPerBlock,
		int partialCount
		)
	{
		int partialId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;
		
		if (partialId < partialCount)
		{
			int weightsPerCell = inputCount + previousOutputCount + 1;

			int cellId = partialId / weightsPerCell;
			int memoryBlockId = cellId / cellsPerBlock;
			int fromId = partialId % weightsPerCell;

			// signal to the cell comes from external input, previous output and bias unit. No if-statements to avoid branch diverging
			int isFromInputUnit = fromId >= 0 && fromId < inputCount;
			int isFromPreviousOutputUnit = (fromId >= inputCount) && (fromId < inputCount + previousOutputCount);
			int isFromBiasUnit = fromId == (inputCount + previousOutputCount);

			float inputFromWeight = isFromInputUnit * input[isFromInputUnit * fromId]
								  + isFromPreviousOutputUnit * previousOutput[isFromPreviousOutputUnit * (fromId - inputCount)]
								  + isFromBiasUnit * 1;

			cellWeightsRTRLPartials[partialId] = cellWeightsRTRLPartials[partialId] * forgetGateActivations[memoryBlockId] + cellInputActivationDerivatives[cellId] * inputGateActivations[memoryBlockId] * inputFromWeight;
		}
	}



	__global__ void LSTMGateWeightsRTRLPartialsKernel(
		float* input,
		float* previousOutput,
		float* previousCellStates,
		float *cellInputActivations,
		float *inputGateActivations,
		float *forgetGateActivations,
		float *inputGateActivationDerivatives,
		float *forgetGateActivationDerivatives,
		float* inputGateWeightsRTRLPartials,
		float* forgetGateWeightsRTRLPartials,

		int inputCount,
		int previousOutputCount,
		int cellsPerBlock,
		int partialCount
		)
	{
		int partialId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;
		
		if (partialId < partialCount)
		{
			int weightsPerGate = inputCount + previousOutputCount + cellsPerBlock + 1;

			int cellId = partialId / weightsPerGate;
			int memoryBlockId = cellId / cellsPerBlock;
			int fromId = partialId % weightsPerGate;

			// signal to the gate comes from external input, previous output, previous cell states and bias unit. No if-statements to avoid branch diverging
			int isFromInputUnit = fromId >= 0 && fromId < inputCount;
			int isFromPreviousOutputUnit = (fromId >= inputCount) && (fromId < inputCount + previousOutputCount);
			int isPeephole = (fromId >= inputCount + previousOutputCount) && (fromId < inputCount + previousOutputCount + cellsPerBlock);
			int isFromBiasUnit = fromId == (inputCount + previousOutputCount + cellsPerBlock);

			float inputFromWeight = isFromInputUnit * input[isFromInputUnit * fromId]
								  + isFromPreviousOutputUnit * previousOutput[isFromPreviousOutputUnit * (fromId - inputCount)]
								  + isPeephole * previousCellStates[isPeephole * (memoryBlockId * cellsPerBlock + (fromId - inputCount - previousOutputCount))]
								  + isFromBiasUnit * 1;

			inputGateWeightsRTRLPartials[partialId] = inputGateWeightsRTRLPartials[partialId] * forgetGateActivations[memoryBlockId] + cellInputActivations[cellId] * inputGateActivationDerivatives[memoryBlockId] * inputFromWeight;
			forgetGateWeightsRTRLPartials[partialId] = forgetGateWeightsRTRLPartials[partialId] * forgetGateActivations[memoryBlockId] + previousCellStates[cellId] * forgetGateActivationDerivatives[memoryBlockId] * inputFromWeight;
		}
	}
}


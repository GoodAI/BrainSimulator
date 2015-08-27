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
	__global__ void LSTMDeltaKernelBPTT(
		float* deltas,
		float* cellStates,
		float* previousCellStates,
		float* cellStateErrors,

		float* outputGateDeltas,
		float* forgetGateDeltas,
		float* inputGateDeltas,
        float* cellInputDeltas,

		float* outputGateActivations,
		float* forgetGateActivations,
		float* inputGateActivations,

		float* cellInputActivationDerivatives,
		float* outputGateActivationDerivatives,
		float* forgetGateActivationDerivatives,
		float* inputGateActivationDerivatives,

		float* cellInputWeights,
		float* outputGateWeights,
		float* forgetGateWeights,
		float* inputGateWeights,

		int inputCount,
		int cellCount,
		int cellsPerBlock
		)
	{
		int memoryBlockId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (memoryBlockId < cellCount / cellsPerBlock)
		{
			float outputGateDeltaSum = 0.0;

			for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
			{
				outputGateDeltaSum += cellStates[cellId] * -deltas[cellId];
			}
			outputGateDeltas[memoryBlockId] = outputGateActivationDerivatives[memoryBlockId] * outputGateDeltaSum;

			for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
			{
				int peepHoleWeightId = (memoryBlockId * (inputCount + cellCount + cellsPerBlock + 1)) + inputCount + cellCount + cellId;
				cellStateErrors[cellId] = -deltas[cellId] * outputGateActivations[memoryBlockId] * cellStates[cellId] +
					cellStateErrors[cellId] * forgetGateActivations[cellId] +
					inputGateDeltas[memoryBlockId] * inputGateWeights[peepHoleWeightId] +
					forgetGateDeltas[memoryBlockId] * forgetGateWeights[peepHoleWeightId] +
					outputGateDeltas[memoryBlockId] * outputGateWeights[peepHoleWeightId];

                cellInputDeltas[cellId] = inputGateActivations[memoryBlockId] *  cellInputActivationDerivatives[memoryBlockId] * cellStateErrors[cellId];
            }

			inputGateDeltas[memoryBlockId] = 0;
			forgetGateDeltas[memoryBlockId] = 0;

			for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
			{
				inputGateDeltas[memoryBlockId] += inputGateActivationDerivatives[memoryBlockId] * cellStateErrors[cellId] * inputGateActivations[memoryBlockId];
				forgetGateDeltas[memoryBlockId] += forgetGateActivationDerivatives[memoryBlockId] * cellStateErrors[cellId] * previousCellStates[cellId];
			}
		}
	}

	__global__ void LSTMGateGradientKernelBPTT(
		float *input,
		float *previousOutput,
		float *cellStates,

		float *inputGateDeltas,
		float *forgetGateDeltas,
		float *outputGateDeltas,

		float* outputGateWeightGradient,
		float* inputGateWeightGradient,
		float* forgetGateWeightGradient,

		int inputCount,
		int previousOutputCount,
		int cellsPerBlock
		)
	{
		int weightId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		int weightsPerGate = inputCount + previousOutputCount + cellsPerBlock + 1;

		if (weightId < weightsPerGate * previousOutputCount / cellsPerBlock)
		{
			int fromId = weightId % weightsPerGate;
			int toId = weightId / weightsPerGate;

			//calculate output gate weight gradient
			int isFromInputUnit = fromId >= 0 && fromId < inputCount;
			int isFromPreviousOutputUnit = (fromId >= inputCount) && (fromId < inputCount + previousOutputCount);
			int isPeephole = (fromId >= inputCount + previousOutputCount) && (fromId < inputCount + previousOutputCount + cellsPerBlock);
			int isFromBiasUnit = fromId == (inputCount + previousOutputCount + cellsPerBlock);

			float inputFromWeight = isFromInputUnit * input[isFromInputUnit * fromId]
				+ isFromPreviousOutputUnit * previousOutput[isFromPreviousOutputUnit * (fromId - inputCount)]
				+ isPeephole * cellStates[isPeephole * (toId * cellsPerBlock + (fromId - inputCount - previousOutputCount))]
				+ isFromBiasUnit * 1;

			outputGateWeightGradient[weightId] = outputGateDeltas[toId] * inputFromWeight;
			inputGateWeightGradient[weightId] = inputGateDeltas[toId] * inputFromWeight;
			forgetGateWeightGradient[weightId] = forgetGateDeltas[toId] * inputFromWeight;
		}
	}

	__global__ void LSTMCellInputGradientKernelBPTT(
		float *input,
		float *previousOutput,

		float *cellInputDeltas,
		float *cellInputWeightGradient,

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
			int fromId = weightId % weightsPerCell;
			int toId = weightId / weightsPerCell;

			int isFromInputUnit = fromId >= 0 && fromId < inputCount;
			int isFromPreviousOutputUnit = (fromId >= inputCount) && (fromId < inputCount + previousOutputCount);
			int isFromBiasUnit = fromId == (inputCount + previousOutputCount);

			float inputFromWeight = isFromInputUnit * input[isFromInputUnit * fromId]
				+ isFromPreviousOutputUnit * previousOutput[isFromPreviousOutputUnit * (fromId - inputCount)]
				+ isFromBiasUnit * 1;

			cellInputWeightGradient[weightId] = cellInputDeltas[toId] * inputFromWeight;
		}
	}

	__global__ void LSTMDeltaBackKernelBPTT(
		ActivationFunctionEnum prevLayerActivationFunction,
		float *prevDeltaPtr,

		float* cellInputDeltas,
		float* outputGateDeltas,
		float* forgetGateDeltas,
		float* inputGateDeltas,

		float *cellInputWeights,
		float *inputGateWeights,
		float *forgetGateWeights,
		float *outputGateWeights,

		int prevLayerNeurons,
		int cellCount,
		int cellsPerBlock
		)
	{
		int neuronId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		int weightsPerCell = prevLayerNeurons + cellCount + 1;
		int weightsPerGate = prevLayerNeurons + cellCount + cellsPerBlock + 1;

		if (neuronId < prevLayerNeurons)
		{
			int memoryBlockCount = cellCount / cellsPerBlock;
			for (int memoryBlockId = 0; memoryBlockId < memoryBlockCount; memoryBlockId++)
			{
				int cellWeightId = memoryBlockId * weightsPerCell + neuronId;
				int gateWeightId = memoryBlockId * weightsPerGate + neuronId;

				prevDeltaPtr[neuronId] += -cellInputDeltas[memoryBlockId] * cellInputWeights[cellWeightId];
				prevDeltaPtr[neuronId] += -inputGateDeltas[memoryBlockId] * inputGateWeights[gateWeightId];
				prevDeltaPtr[neuronId] += -forgetGateDeltas[memoryBlockId] * forgetGateWeights[gateWeightId];
				prevDeltaPtr[neuronId] += -outputGateDeltas[memoryBlockId] * outputGateWeights[gateWeightId];
			}
		}
	}



	/*****************************************************************************************************************************************************************/
	/*****************************************************************************************************************************************************************/
	/*****************************************************************************************************************************************************************/
	/*****************************************************************************************************************************************************************/
    /*
    /*  ORIGINAL FROM KAREL
     */
	/*****************************************************************************************************************************************************************/
	/*****************************************************************************************************************************************************************/
	/*****************************************************************************************************************************************************************/



	__global__ void LSTMDeltaKernel(
		float *cellStateErrors,
		float *outputGateDeltas,
		float *cellStates,
		float *outputGateActivations,
		float *outputGateActivationDerivatives,
		float *deltas,

		int cellCount,
		int cellsPerBlock
		)
	{
		int memoryBlockId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (memoryBlockId < cellCount / cellsPerBlock)
		{
			float outputGateDeltaSum = 0.0;

			for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
			{
				float delta = -deltas[cellId];
				cellStateErrors[cellId] = outputGateActivations[memoryBlockId] * delta;
				outputGateDeltaSum += cellStates[cellId] * delta;
			}

			outputGateDeltas[memoryBlockId] = outputGateActivationDerivatives[memoryBlockId] * outputGateDeltaSum;
		}
	}

	__global__ void LSTMDeltaBackKernel(
		ActivationFunctionEnum prevLayerActivationFunction,
		float *prevWeighedInputPtr,
		float *prevDeltaPtr,
		float *cellStateErrors,
		float *previousCellStates,
		float *inputGateActivations,

		float *cellInputActivationDerivatives,
		float *inputGateActivationDerivatives,
		float *forgetGateActivationDerivatives,

		float *cellInputWeights,
		float *inputGateWeights,
		float *forgetGateWeights,
		float *outputGateWeights,

		float *outputGateDeltas,

		int prevLayerNeurons,
		int cellCount,
		int cellsPerBlock
		)
	{
		int neuronId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (neuronId < prevLayerNeurons)
		{
			float delta = 0.0f;

			for (int memoryBlockId = 0; memoryBlockId < cellCount / cellsPerBlock; memoryBlockId++)
			{
				float inputGateError = 0.0f;
				float forgetGateError = 0.0f;

				for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
				{
					inputGateError += inputGateActivationDerivatives[memoryBlockId] * cellStateErrors[cellId] * inputGateActivations[memoryBlockId];
					forgetGateError += forgetGateActivationDerivatives[memoryBlockId] * cellStateErrors[cellId] * previousCellStates[cellId];
					// cell input error
					delta += cellInputWeights[cellId * (prevLayerNeurons + cellCount + 1) + neuronId] * inputGateActivations[memoryBlockId] * cellStateErrors[cellId] * cellInputActivationDerivatives[cellId];
				}

				delta += inputGateWeights[memoryBlockId * (prevLayerNeurons + cellCount + cellsPerBlock + 1) + neuronId] * inputGateError;
				delta += forgetGateWeights[memoryBlockId * (prevLayerNeurons + cellCount + cellsPerBlock + 1) + neuronId] * forgetGateError;
				delta += outputGateWeights[memoryBlockId * (prevLayerNeurons + cellCount + cellsPerBlock + 1) + neuronId] * outputGateDeltas[memoryBlockId];
			}

			prevDeltaPtr[neuronId] = -delta * EvaluateDerivative(prevLayerActivationFunction, prevWeighedInputPtr[neuronId]);
		}
	}
 
  
 

}

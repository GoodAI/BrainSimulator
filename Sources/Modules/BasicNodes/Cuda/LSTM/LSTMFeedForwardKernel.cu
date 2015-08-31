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
	__device__ float Clip(float value, float clip)
	{
		return (clip == 0) * value + (clip != 0) * ((value > clip) * clip + (value < -clip) * -clip + (value >= -clip && value <= clip) * value);

		/* avoids thread divergence, equivalent to:
		if (clip == 0)
			return value;
		else if (value > clip)
			return clip;
		else if (value < -clip)
			return -clip;
		else
			return value;
		*/
	}

	__device__ float GetNetInput(
		int memoryBlockId,
		int cellsPerBlock,
		float* weights,
		int weightsOffset,
		float *input,
		int inputCount,
		float *previousOutput,
		int previousOutputCount,
		float *cellStates,
		bool peephole,
		bool bias
		)
	{
		int weightId = weightsOffset;
		float netInput = 0;

		// signal from external input
		for (int i = 0; i < inputCount; i++)
		{
			netInput += weights[weightId] * input[i];
			weightId++;
		}

		// signal from previous output of memory blocks
		for (int i = 0; i < previousOutputCount; i++)
		{
			netInput += weights[weightId] * previousOutput[i];
			weightId++;
		}

		// signal from peephole connections
		if (peephole)
		{
			for (int i = 0; i < cellsPerBlock; i++)
			{
				netInput += weights[weightId] * cellStates[memoryBlockId * cellsPerBlock + i];
				weightId++;
			}
		}

		if (bias)
		{
			netInput += weights[weightId];
		}

		return netInput;
	}


	__global__ void LSTMFeedForwardKernelBPTT(
		ActivationFunctionEnum inputActivationFunction,
		ActivationFunctionEnum gateActivationFunction,
		float *input,
		float *output,
		float *previousOutput,
		float *cellStates,
		float *cellStatesActivations,
		float *cellStateActivationDerivatives,
		float *previousCellStates,
		float *cellInputActivations,
		float *cellInputActivationDerivatives,
		float *inputGateActivations,
		float *inputGateActivationDerivatives,
		float *forgetGateActivations,
		float *forgetGateActivationDerivatives,
		float *outputGateActivations,
		float *outputGateActivationDerivatives,

		float *cellInputWeights,
		float *inputGateWeights,
		float *forgetGateWeights,
		float *outputGateWeights,

		float clipCellState,

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

			// step 1: calculate activations of input and forget gate
			float inputGateNetInput = GetNetInput(
				memoryBlockId,
				cellsPerBlock,
				inputGateWeights,
				memoryBlockId * (inputCount + cellCount + cellsPerBlock + 1),
				input,
				inputCount,
				previousOutput,
				cellCount,
				previousCellStates,
				true,
				true
			);
			float forgetGateNetInput = GetNetInput(
				memoryBlockId,
				cellsPerBlock,
				forgetGateWeights,
				memoryBlockId * (inputCount + cellCount + cellsPerBlock + 1),
				input,
				inputCount,
				previousOutput,
				cellCount,
				previousCellStates,
				true,
				true
			);

			// activation function of all gates must be in range [0,1], sigmoid activation function is used
			float inputGateActivation = Evaluate(gateActivationFunction, inputGateNetInput);
			float forgetGateActivation = Evaluate(gateActivationFunction, forgetGateNetInput);

			inputGateActivations[memoryBlockId] = inputGateActivation;
			forgetGateActivations[memoryBlockId] = forgetGateActivation;

			inputGateActivationDerivatives[memoryBlockId] = EvaluateDerivative(gateActivationFunction, inputGateNetInput);
			forgetGateActivationDerivatives[memoryBlockId] = EvaluateDerivative(gateActivationFunction, forgetGateNetInput);

			// step 2: calculate activation of memory block's cells
			for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
			{
				float cellNetInput = GetNetInput(
					memoryBlockId,
					cellsPerBlock,
					cellInputWeights,
					cellId * (inputCount + cellCount + 1),
					input,
					inputCount,
					previousOutput,
					cellCount,
					NULL,
					false,
					true
				);

				float cellInputActivation = Evaluate(inputActivationFunction, cellNetInput);

				cellInputActivations[cellId] = cellInputActivation;
				cellInputActivationDerivatives[cellId] = EvaluateDerivative(inputActivationFunction, cellNetInput);

				cellStates[cellId] = Clip(forgetGateActivation * previousCellStates[cellId] + inputGateActivation * cellInputActivation, clipCellState);
			}

			// step 3: calculate output gate activation
			float outputGateNetInput = GetNetInput(
				memoryBlockId,
				cellsPerBlock,
				outputGateWeights,
				memoryBlockId * (inputCount + cellCount + cellsPerBlock + 1),
				input,
				inputCount,
				previousOutput,
				cellCount,
				cellStates,
				true,
				true
			);

			float outputGateActivation = Evaluate(gateActivationFunction, outputGateNetInput);
			outputGateActivations[memoryBlockId] = outputGateActivation;
			outputGateActivationDerivatives[memoryBlockId] = EvaluateDerivative(gateActivationFunction, outputGateNetInput);

			// step 4: calculate output of all memory block's cells
			for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
			{
				cellStatesActivations[cellId] = Evaluate(inputActivationFunction, cellStates[cellId]);
				cellStateActivationDerivatives[cellId] = EvaluateDerivative(inputActivationFunction, cellStates[cellId]);

				output[cellId] = outputGateActivation * cellStatesActivations[cellId];
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



	__global__ void LSTMFeedForwardKernel(
		ActivationFunctionEnum inputActivationFunction,
		ActivationFunctionEnum gateActivationFunction,
		float *input,
		float *output,
		float *previousOutput,
		float *cellStates,
		float *previousCellStates,
		float *cellInputActivations,
		float *cellInputActivationDerivatives,
		float *inputGateActivations,
		float *inputGateActivationDerivatives,
		float *forgetGateActivations,
		float *forgetGateActivationDerivatives,
		float *outputGateActivations,
		float *outputGateActivationDerivatives,

		float *cellInputWeights,
		float *inputGateWeights,
		float *forgetGateWeights,
		float *outputGateWeights,

		float clipCellState,

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

			// step 1: calculate activations of input and forget gate
			float inputGateNetInput = GetNetInput(
				memoryBlockId,
				cellsPerBlock,
				inputGateWeights,
				memoryBlockId * (inputCount + cellCount + cellsPerBlock + 1),
				input,
				inputCount,
				previousOutput,
				cellCount,
				previousCellStates,
				true,
				true
			);
			float forgetGateNetInput = GetNetInput(
				memoryBlockId,
				cellsPerBlock,
				forgetGateWeights,
				memoryBlockId * (inputCount + cellCount + cellsPerBlock + 1),
				input,
				inputCount,
				previousOutput,
				cellCount,
				previousCellStates,
				true,
				true
			);

			// activation function of all gates must be in range [0,1], sigmoid activation function is used
			float inputGateActivation = Evaluate(gateActivationFunction, inputGateNetInput);
			float forgetGateActivation = Evaluate(gateActivationFunction, forgetGateNetInput);

			inputGateActivations[memoryBlockId] = inputGateActivation;
			forgetGateActivations[memoryBlockId] = forgetGateActivation;

			inputGateActivationDerivatives[memoryBlockId] = EvaluateDerivative(gateActivationFunction, inputGateNetInput);
			forgetGateActivationDerivatives[memoryBlockId] = EvaluateDerivative(gateActivationFunction, forgetGateNetInput);

			// step 2: calculate activation of memory block's cells
			for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
			{
				float cellNetInput = GetNetInput(
					memoryBlockId,
					cellsPerBlock,
					cellInputWeights,
					cellId * (inputCount + cellCount + 1),
					input,
					inputCount,
					previousOutput,
					cellCount,
					NULL,
					false,
					true
				);

				float cellInputActivation = Evaluate(inputActivationFunction, cellNetInput);

				cellInputActivations[cellId] = cellInputActivation;
				cellInputActivationDerivatives[cellId] = EvaluateDerivative(inputActivationFunction, cellNetInput);

				cellStates[cellId] = Clip(forgetGateActivation * previousCellStates[cellId] + inputGateActivation * cellInputActivation, clipCellState);
			}

			// step 3: calculate output gate activation
			float outputGateNetInput = GetNetInput(
				memoryBlockId,
				cellsPerBlock,
				outputGateWeights,
				memoryBlockId * (inputCount + cellCount + cellsPerBlock + 1),
				input,
				inputCount,
				previousOutput,
				cellCount,
				cellStates,
				true,
				true
			);

			float outputGateActivation = Evaluate(gateActivationFunction, outputGateNetInput);
			outputGateActivations[memoryBlockId] = outputGateActivation;
			outputGateActivationDerivatives[memoryBlockId] = EvaluateDerivative(gateActivationFunction, outputGateNetInput);

			// step 4: calculate output of all memory block's cells
			for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
			{
				output[cellId] = outputGateActivation * cellStates[cellId];
			}
		}
	}
}

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

#include "../Common/Reduction/Reduction.cu"

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

	__global__ void GetNetInput(
		float* netInput,
		float* temporary,
		int cellsPerBlock,
		float* weights,
		float* input,
		int inputCount,
		float* previousOutput,
		int previousOutputCount,
		float* cellStates,
		int peephole,
		int bias
		)
	{
		const int THREAD_CNT = 512;

		int size = inputCount + previousOutputCount + peephole * cellsPerBlock + bias;
		int memoryBlockId = blockIdx.x;
		int blockOffset = memoryBlockId * size;
		int tid = threadIdx.x;

		int weightOffset = blockOffset;

		// signal from external input
		for (int i = tid; i < inputCount; i += THREAD_CNT)
		{
			temporary[weightOffset + i] = weights[weightOffset + i] * input[i];
		}
		weightOffset += inputCount;

		//// signal from previous output of memory blocks
		for (int i = tid; i < previousOutputCount; i += THREAD_CNT)
		{
			temporary[weightOffset + i] = weights[weightOffset + i] * previousOutput[i];
		}
		weightOffset += previousOutputCount;

		// signal from peephole connections
		if (peephole)
		{
			for (int i = tid; i < cellsPerBlock; i += THREAD_CNT)
			{
				temporary[weightOffset + i] = weights[weightOffset + i] * cellStates[memoryBlockId * cellsPerBlock + i];
			}
			weightOffset += cellsPerBlock;
		}

		if (bias)
		{
			temporary[weightOffset] = weights[weightOffset];
		}

		DReduction<f_Sum_f, float, THREAD_CNT>((void*)netInput, (void*)temporary, nullptr, size, memoryBlockId, memoryBlockId * size, 1, true);
	}

	__global__ void CellStateFeedForwardKernelBPTT(
		ActivationFunctionEnum inputActivationFunction,
		ActivationFunctionEnum gateActivationFunction,

		float* previousCellStates,

		float* cellStates,
		float* cellStatesActivations,
		float* cellStateActivationDerivatives,

		float* cellInputNetInput,
		float* cellInputActivations,
		float* cellInputActivationDerivatives,

		float* inputGateNetInput,
		float* inputGateActivations,
		float* inputGateActivationDerivatives,

		float* forgetGateNetInput,
		float* forgetGateActivations,
		float* forgetGateActivationDerivatives,

		int cellCount,
		int cellsPerBlock,
		float clipCellState
		)
	{
		int memoryBlockId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (memoryBlockId < cellCount / cellsPerBlock)
		{
			// activation function of all gates must be in range [0,1], sigmoid activation function is used
			float inputGateActivation = Evaluate(gateActivationFunction, inputGateNetInput[memoryBlockId]);
			inputGateActivations[memoryBlockId] = inputGateActivation;
			inputGateActivationDerivatives[memoryBlockId] = EvaluateDerivative(gateActivationFunction, inputGateNetInput[memoryBlockId]);

			float forgetGateActivation = Evaluate(gateActivationFunction, forgetGateNetInput[memoryBlockId]);
			forgetGateActivations[memoryBlockId] = forgetGateActivation;
			forgetGateActivationDerivatives[memoryBlockId] = EvaluateDerivative(gateActivationFunction, forgetGateNetInput[memoryBlockId]);

			// step 2: calculate activation of memory block's cells
			for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
			{
				float cellInputActivation = Evaluate(inputActivationFunction, cellInputNetInput[cellId]);

				cellInputActivations[cellId] = cellInputActivation;
				cellInputActivationDerivatives[cellId] = EvaluateDerivative(inputActivationFunction, cellInputNetInput[cellId]);

				cellStates[cellId] = Clip(forgetGateActivation * previousCellStates[cellId] + inputGateActivation * cellInputActivation, clipCellState);

				cellStatesActivations[cellId] = Evaluate(inputActivationFunction, cellStates[cellId]);
				cellStateActivationDerivatives[cellId] = EvaluateDerivative(inputActivationFunction, cellStates[cellId]);
			}
		}
	}

	__global__ void OutputStateFeedForwardKernelBPTT(
		ActivationFunctionEnum gateActivationFunction,
		
		float* cellStatesActivations,

		float* output,
		float* outputGateNetInput,
		float* outputGateActivations,
		float* outputGateActivationDerivatives,

		int cellCount,
		int cellsPerBlock,
		float clipCellState
		)
	{
		int memoryBlockId = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (memoryBlockId < cellCount / cellsPerBlock)
		{
			// step 3: calculate output gate activation
			float outputGateActivation = Evaluate(gateActivationFunction, outputGateNetInput[memoryBlockId]);
			outputGateActivations[memoryBlockId] = outputGateActivation;
			outputGateActivationDerivatives[memoryBlockId] = EvaluateDerivative(gateActivationFunction, outputGateNetInput[memoryBlockId]);

			// step 4: calculate output of all memory block's cells
			for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
			{
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


	__device__ float DeviceGetNetInput(
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

	__global__ void LSTMFeedForwardKernel(
		ActivationFunctionEnum inputActivationFunction,
		ActivationFunctionEnum gateActivationFunction,
		ActivationFunctionEnum activationFunction,
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
			float inputGateNetInput = DeviceGetNetInput(
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
			float forgetGateNetInput = DeviceGetNetInput(
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
				float cellNetInput = DeviceGetNetInput(
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
			float outputGateNetInput = DeviceGetNetInput(
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
				output[cellId] = outputGateActivation * cellStates[cellId]; //Evaluate(activationFunction, cellStates[cellId]);
			}
		}
	}
}

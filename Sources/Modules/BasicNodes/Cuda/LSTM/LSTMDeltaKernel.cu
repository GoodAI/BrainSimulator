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



__global__ void LSTMDeltasBackKernel(
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

        float *inputGateDeltas, // ADDED
        float *forgetGateDeltas, //ADDED
        float *cellInputDeltas, // Painter:ADDED

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
                    cellInputDeltas[cellId] = cellInputWeights[cellId * (prevLayerNeurons + cellCount + 1) + neuronId] * inputGateActivations[memoryBlockId] * cellStateErrors[cellId] * cellInputActivationDerivatives[cellId];
					delta += cellInputDeltas[cellId];
				}
                inputGateDeltas[memoryBlockId]  = inputGateError;
                forgetGateDeltas[memoryBlockId] = forgetGateError;

				delta += inputGateWeights[memoryBlockId * (prevLayerNeurons + cellCount + cellsPerBlock + 1) + neuronId] * inputGateError;
				delta += forgetGateWeights[memoryBlockId * (prevLayerNeurons + cellCount + cellsPerBlock + 1) + neuronId] * forgetGateError;
				delta += outputGateWeights[memoryBlockId * (prevLayerNeurons + cellCount + cellsPerBlock + 1) + neuronId] * outputGateDeltas[memoryBlockId];
			}

			prevDeltaPtr[neuronId] = -delta * EvaluateDerivative(prevLayerActivationFunction, prevWeighedInputPtr[neuronId]);
		}
	}


}

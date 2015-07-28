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
		float *inputGateActivations,
		float *cellInputWeights,

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
			float deltaSum = 0.0f;

			for (int memoryBlockId = 0; memoryBlockId < cellCount / cellsPerBlock; memoryBlockId++)
			{
				float inputGate = inputGateActivations[memoryBlockId];
				for (int cellId = memoryBlockId * cellsPerBlock; cellId < (memoryBlockId + 1) * cellsPerBlock; cellId++)
				{
					deltaSum += cellInputWeights[cellId * (prevLayerNeurons + cellCount + 1) + neuronId] * inputGate * cellStateErrors[cellId];
				}
			}

			prevDeltaPtr[neuronId] = -deltaSum * EvaluateDerivative(prevLayerActivationFunction, prevWeighedInputPtr[neuronId]);
		}
	}
}

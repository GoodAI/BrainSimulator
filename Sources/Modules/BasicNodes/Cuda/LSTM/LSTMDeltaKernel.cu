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
		float *nextLayerDeltas,
		float *nextLayerWeights,

		int nextLayerSize,
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
				float sum = 0.0;
				int index = memoryBlockId * nextLayerSize;
				for (int j = 0; j < nextLayerSize; j++)
				{
					sum += nextLayerWeights[index + j] * nextLayerDeltas[j];
				}
				cellStateErrors[cellId] = outputGateActivations[memoryBlockId] * sum;
				outputGateDeltaSum += cellStates[cellId] * sum;
			}

			outputGateDeltas[memoryBlockId] = outputGateActivationDerivatives[memoryBlockId] * outputGateDeltaSum;
		}
	}
}

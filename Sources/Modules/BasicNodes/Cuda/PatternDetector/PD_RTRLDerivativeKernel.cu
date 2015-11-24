//Includes for IntelliSense 
#define _SIZE_T_DEFINED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math.h>


extern "C"  
{
	__constant__ int D_INPUT_UNITS;
	__constant__ int D_HIDDEN_UNITS;
	__constant__ int D_OUTPUT_UNITS;


	__global__ void InputWeightsRTRLDerivativesKernel(
		float *input,
		float *hiddenActivationDerivatives,
		float *recurrentWeights,
		float *inputWeightRTRLDerivatives,
		float *previousInputWeightRTRLDerivatives
		)
	{
		int partialId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (partialId < D_HIDDEN_UNITS * D_HIDDEN_UNITS * D_INPUT_UNITS)
		{
			int unitId = partialId / (D_HIDDEN_UNITS * D_INPUT_UNITS);
			int weightId = partialId % (D_HIDDEN_UNITS * D_INPUT_UNITS);
			int to = weightId / D_INPUT_UNITS;
			int from = weightId % D_INPUT_UNITS;
			
			float sum = 0;
			for (int i = 0; i < D_HIDDEN_UNITS; i++)
			{
				sum += recurrentWeights[unitId * D_HIDDEN_UNITS + i] * previousInputWeightRTRLDerivatives[i * (D_HIDDEN_UNITS * D_INPUT_UNITS) + weightId];
			}

			inputWeightRTRLDerivatives[partialId] = hiddenActivationDerivatives[unitId] * ((unitId == to) * input[from] + sum);
		}
	}

	__global__ void RecurrentWeightsRTRLDerivativesKernel(
		float *previousHiddenActivations,
		float *hiddenActivationDerivatives,
		float *recurrentWeights,
		float *recurrentWeightRTRLDerivatives,
		float *previousRecurrentWeightRTRLDerivatives
		)
	{
		int partialId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (partialId < D_HIDDEN_UNITS * D_HIDDEN_UNITS * D_HIDDEN_UNITS)
		{
			int unitId = partialId / (D_HIDDEN_UNITS * D_HIDDEN_UNITS);
			int weightId = partialId % (D_HIDDEN_UNITS * D_HIDDEN_UNITS);
			int to = weightId / D_HIDDEN_UNITS;
			int from = weightId % D_HIDDEN_UNITS;

			float sum = 0;
			for (int i = 0; i < D_HIDDEN_UNITS; i++)
			{
				sum += recurrentWeights[unitId * D_HIDDEN_UNITS + i] * previousRecurrentWeightRTRLDerivatives[i * (D_HIDDEN_UNITS * D_HIDDEN_UNITS) + weightId];
			}

			recurrentWeightRTRLDerivatives[partialId] = hiddenActivationDerivatives[unitId] * ((unitId == to) * previousHiddenActivations[from] + sum);
		}
	}
}

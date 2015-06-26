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
	__constant__ int D_UNITS;
	__constant__ int D_HIDDEN_UNIT_WEIGHTS;
	__constant__ int D_OUTPUT_UNIT_WEIGHTS;
	__constant__ int D_WEIGHTS;
	

	//kernel code
	__global__ void RTRLDerivativeKernel(
		float *activation,
		float *previousActivation,
		float *activationDerivative,
		float *weights,
		float *RTRLDerivatives,
		float *previousRTRLDerivatives
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (threadId < D_WEIGHTS * (D_HIDDEN_UNITS + D_OUTPUT_UNITS))
		{
			int unitId = threadId % (D_HIDDEN_UNITS + D_OUTPUT_UNITS);
			int isHiddenUnit = unitId < D_HIDDEN_UNITS;
			//rows to be skipped in the weights matrix
			int offset = isHiddenUnit * (unitId * (1 + D_INPUT_UNITS + D_HIDDEN_UNITS)); //hidden units are connected to threshold, input and hidden units
			offset += !isHiddenUnit * (D_HIDDEN_UNIT_WEIGHTS //if output unit then skip all hidden unit weights
					+ (unitId - D_HIDDEN_UNITS) * (1 + D_HIDDEN_UNITS)); //output units are connected to threshold and hidden units

			int weightId = threadId / (D_HIDDEN_UNITS + D_OUTPUT_UNITS);
			int isHiddenUnitWeight = weightId < D_HIDDEN_UNIT_WEIGHTS;

			int to = (weightId - !isHiddenUnitWeight * D_HIDDEN_UNIT_WEIGHTS) / (1 + isHiddenUnitWeight * D_INPUT_UNITS + D_HIDDEN_UNITS);
			int from = (weightId - !isHiddenUnitWeight * D_HIDDEN_UNIT_WEIGHTS) % (1 + isHiddenUnitWeight * D_INPUT_UNITS + D_HIDDEN_UNITS);
			from += !isHiddenUnitWeight * (from > 0) * D_INPUT_UNITS; //if this is weight from hidden unit to output unit, skip threshold and input units

			float weightedSum = 0.0f;

			for (int i = 0; i < D_HIDDEN_UNITS; i++)
			{
				weightedSum += weights[offset + 1 + isHiddenUnit * D_INPUT_UNITS + i] * previousRTRLDerivatives[weightId * (D_HIDDEN_UNITS + D_OUTPUT_UNITS) + i];
			}

			RTRLDerivatives[threadId] = activationDerivative[1 + D_INPUT_UNITS + unitId] * (weightedSum + (to == unitId) * activation[from]);
		}
	}
}

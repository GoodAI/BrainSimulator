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
	__global__ void FullyConnectedCurvatureKernel(
		float *weightsGradPtr,
		float *biasGradPtr,
		float *shiftedWeightsPtr,
		float *shiftedBiasPtr,
		float *avgWeightGradPtr,
		float *avgBiasGradPtr,
		float *weightGradCurvePtr,
		float *biasGradCurvePtr,
		float *dropoutMaskPtr,
		int prevLayerSize,
		int thisLayerSize
		)
	{
		// i: prev. layer neuron id
		// j: current layer neuron id
		float avgGrad;
		int i;
		int j = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (j < thisLayerSize)
		{
			if (!dropoutMaskPtr[j])
			{
				int index = j;
				for (i = 0; i < prevLayerSize; i++)
				{
					// weight finite difference curvature
					avgGrad = avgWeightGradPtr[index];
					if (avgGrad == 0)
						avgGrad == 0.000001; // don't divide by 0!
					weightGradCurvePtr[index] = abs(weightsGradPtr[index] - shiftedWeightsPtr[index]) / avgGrad;
					index += thisLayerSize;
				}

				// bias finite difference curvature
				avgGrad = avgBiasGradPtr[j];
				if (avgGrad == 0)
					avgGrad == 0.000001; // don't divide by 0!
				biasGradCurvePtr[j] = abs(biasGradPtr[j] - shiftedBiasPtr[j]) / avgGrad;
			}
		}
	}
}
#define _SIZE_T_DEFINED 

#include <cuda.h> 
#include <math_constants.h> 
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_types.h> 
#include <vector_functions.h> 
#include <float.h>

#include "ColorHelpers.cu"

extern "C"
{
	__global__ void ColorScaleObserverComplex(float* values, int method, int scale, float minValue, float maxValue, unsigned int* pixels, int numOfPixels)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		if (id < numOfPixels) //id of the thread is valid
		{
			pixels[id] = float_to_uint_rgba(values[id], method, scale, minValue, maxValue);
		}
	}
}
#define _SIZE_T_DEFINED 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __cplusplus 
#define __cplusplus 
#endif

#include <cuda.h> 
#include <math_constants.h> 
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_types.h> 
#include <vector_functions.h> 



extern "C"  
{	
	//kernel code
	__global__ void PlotObserverScaleDownScaleKernel(float* history, int nbCurves, int size)
	{		
		int id = blockDim.x*blockIdx.y*gridDim.x	
				+ blockDim.x*blockIdx.x				
				+ threadIdx.x;

		if (id >= size)
			return;
		
		int baseAddress = 2 * id;
		float val1 = history[baseAddress];
		float val2 = history[baseAddress + nbCurves];
		float average = (val1 + val2) / 2;
		history[id] = average;
	}
}
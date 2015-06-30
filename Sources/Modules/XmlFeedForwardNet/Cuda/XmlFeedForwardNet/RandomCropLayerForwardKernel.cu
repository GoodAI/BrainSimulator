#define _SIZE_T_DEFINED 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __cplusplus 
#define __cplusplus 
#endif

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>

#include "utils.cu"


extern "C"  
{
	// 1 thread = 1 pixel

	//kernel code
	__global__ void RandomCropLayerForwardKernel(
									int		ShiftX,
									int		ShiftY,
									MyLayerDim* OutputDataPtr,
									MyLayerDim* InputDataPtr
									)
	{
		int outputId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
		
		__shared__ MyLayerDim s_input;
		__shared__ MyLayerDim s_output;

		if (threadIdx.x == 0)
		{
			s_input = *InputDataPtr;
			s_output = *OutputDataPtr;
		}

		__syncthreads();

		if(outputId < s_output.Count)
		{
			int outputZ = outputId / s_output.Size;
			int outputY = (outputId - outputZ * s_output.Size) / s_output.Width;
			int outputX = outputId - outputZ * s_output.Size - outputY * s_output.Width;

			int inputZ = outputZ;
			int inputY = outputY + ShiftY;
			int inputX = outputX + ShiftX;
		
			float outputValue = s_input.Ptr[inputX + inputY * s_input.Width + inputZ * s_input.Size];
		
			s_output.Ptr[outputId] = outputValue;
		}
	}
}
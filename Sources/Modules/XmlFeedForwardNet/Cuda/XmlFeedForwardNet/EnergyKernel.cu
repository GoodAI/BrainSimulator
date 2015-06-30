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


extern "C"  
{
	// TODO – replace this kernel (only called once on one thread...)
	// Only one thread

	__global__ void EnergyKernel(
							unsigned int energyBufferSize,
							float*	networkOutput,
							float*	label,
							int		labelSize,
							float*  currentSampleEnergy,
							float*	energySample,
							float*	energy
							)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
					+ blockDim.x * blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		if(id > 0)
			return;
		
		double sum = 0;
		for (int i = 0; i < labelSize; i++)
		{
			double diff = label[i] - networkOutput[i];
			sum += diff * diff;
		}
		*currentSampleEnergy = sum;


		double averageSum = 0;
		for (int i = 0; i < energyBufferSize; i++)
		{
			averageSum += energySample[i];
		}
		energy[0] = averageSum / energyBufferSize;
	}
}

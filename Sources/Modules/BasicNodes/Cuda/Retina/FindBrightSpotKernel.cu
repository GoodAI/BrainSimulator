#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>

extern "C"  
{	
	__global__ void ApplyEyeMovement(float* currentEye, float* stats, float moveFactor, float scaleFactor, float scaleBase) 
	{		
		float sumWeights = stats[4];

		if (sumWeights > 0) 
		{
			currentEye[0] = fmaxf(fminf(moveFactor * stats[0], 1), -1);
			currentEye[1] = fmaxf(fminf(moveFactor * stats[1], 1), -1);

			float variance = sqrtf((stats[2] + stats[3]) * 0.5);

			currentEye[2] = fmaxf(fminf(variance * scaleFactor + scaleBase, 1), 0);
		}
		else 
		{
			currentEye[0] = 0;
			currentEye[1] = 0;
			currentEye[2] = 1;
		}
	}

	/*
	__global__ void ApplyEyeMovementK_Means(float* currentEye, float* focusedCentroid, float scaleFactor, float scaleBase) 
	{		
		float sumWeights = stats[4];

		if (sumWeights > 0) 
		{
			currentEye[0] = fmax(fmin(focusedCentroid[0], 1), -1);
			currentEye[1] = fmax(fmin(focusedCentroid[1], 1), -1);

			float variance = sqrtf((stats[2] + stats[3]) * 0.5);

			currentEye[2] = fmax(fmin(variance * scaleFactor + scaleBase, 1), 0);
		}
		else 
		{
			currentEye[0] = 0;
			currentEye[1] = 0;
			currentEye[2] = 1;
		}
	}
	*/
}
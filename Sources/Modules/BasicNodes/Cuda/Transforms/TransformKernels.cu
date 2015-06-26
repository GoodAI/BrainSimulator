#define _SIZE_T_DEFINED

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>



extern "C"  
{		
	__global__ void AbsoluteValueKernel(float* input, float* output, int size)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		if(id < size)
		{	
			output[id] = fabsf(input[id]);
		}
	}		

	__global__ void ModuloKernel(float* input, int divisor, float* output, int size)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x	+ blockDim.x * blockIdx.x	+ threadIdx.x;
		if(id < size)
		{
			output[id] = (float)   (((int)input[id]) % divisor) ;
		}
	}
	__global__ void RoundKernel(float* input, float* output, int size)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x	+ blockDim.x * blockIdx.x	+ threadIdx.x;
		if(id < size)
		{
			output[id] = round(input[id]);
		}
	}
	__global__ void FloorKernel(float* input, float* output, int size)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x	+ blockDim.x * blockIdx.x	+ threadIdx.x;
		if(id < size)
		{
			output[id] = floor(input[id]);
		}
	}	
	__global__ void CeilKernel(float* input, float* output, int size)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x	+ blockDim.x * blockIdx.x	+ threadIdx.x;
		if(id < size)
		{
			output[id] = ceil(input[id]);
		}
	}	



	__global__ void CropKernel(float min, float max, float* input, float* output, int size)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		if(id < size)
		{	
			output[id] = fmaxf(fminf(input[id], max), min);
		}
	}	

	__global__ void ThresholdKernel(float min, float max, float* input, float* output, int size, int count)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		__shared__ float delta;

		if(threadIdx.x < size)
		{	
			if (id == 0)
				delta = (max - min)/count;
			__syncthreads();

			for (int i = 0; i < count; i++)
				output[id * count + i] = 0;

			int idx = (int)floor(fmaxf(0, fminf(((input[id] - min) / delta), count - 1)));
			//			output[id * count + idx] = 1.0f;
			output[idx * size + id] = 1.0f;
		}
	}	


	__global__ void PolynomialFunctionKernel(float a3, float a2, float a1, float a0, float* input, float* output, int size)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		if(id < size)
		{	
			float x = input[id];
			output[id] = a3 * x * x * x + a2 * x * x + a1 * x + a0;
		}
	}	

	__global__ void PolynomialFunctionKernel_Double(float a3, float a2, float a1, float a0, double* input, double* output, int size)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		if (id < size)
		{
			double x = input[id];
			output[id] = a3 * x * x * x + a2 * x * x + a1 * x + a0;
		}
	}

	__global__ void LinearFunctionKernelDouble(double a1, double a0, double* input, double* output, int size)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		if(id < size)
		{	
			double x = input[id];
			output[id] =  a1 * x + a0;
		}
	}	

	__global__ void LinearValuesKernel(const float min, const float max, float* output, const int size, const int shift)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		__shared__ float delta;

		if (threadIdx.x == 0) 
			delta = (max-min)/(size-1);
		__syncthreads();

		if(id < size)
		{				
			output[(id + shift) % size] = min + id * delta;
		}
	}	

	__global__ void GoniometricFunctionKernel(float* input, float* output, const int size, const int type)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;
		if(id < size)
		{	 // Sine = 0, Cosine = 1, Tan = 2, Tanh = 3, Sinh = 4, Cosh = 5  see MyGonioType in MyTransform.cs
			switch (type) 
			{
			case 0:
				output[id] = sinf(input[id]);
				break;
			case 1:
				output[id] = cosf(input[id]);
				break;
			case 2:
				output[id] = tanf(input[id]);
				break;
			case 3:
				output[id] = tanhf(input[id]);
				break;
			case 4:
				output[id] = sinhf(input[id]);
				break;
			case 5:
				output[id] = coshf(input[id]);
				break;			
			}
		}		
	}

	__global__ void AddAndApproachValueKernel(float targetValue, float delta, float factor, int method, float* input, float* output, int size)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		if(id < size)
		{	           
			if (method == 0) 
			{
				output[id] += input[id];
				float x = output[id];

				float diff = targetValue - x;
				float finalDelta = copysign(delta, diff);
				output[id] = x + finalDelta;
			}
			else if (method == 1) 
			{
				output[id] += input[id];
				float x = output[id];

				output[id] = factor * (x - targetValue) + targetValue;
			}
			else 
			{
				output[id] = input[id] * (1 - factor) + output[id] * factor;
			}
		}
	}	

	__global__ void LinearCombinationKernel(float *input1, float input1_coeff, int input1_start_index, float *input2, float input2_coeff, int input2_start_index, float *output, int output_start_index, int size)
	{		
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		if(id < size)
		{
			output[output_start_index + id] = input1_coeff * input1[input1_start_index + id] + input2_coeff * input2[input2_start_index + id];
		}
	}

	
    typedef enum eMyComparaisonFunction
    {
        EQUAL_TO =                  0,
        DIFFERENT_TO  =             1,
        LESS_THAN =                 2,
        LESS_THAN_OR_EQUAL_TO =     3,
        GREATER_THAN =              4,
        GREATER_THAN_OR_EQUAL_TO =  5
    } MyComparaisonFunction;

	__global__ void CompareToSingle(float* input, float* output, eMyComparaisonFunction method, float value, int count)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		if (id < count)
		{
			float inputValue = input[id];
			bool result;
			switch (method)
			{
				case EQUAL_TO:
					result = (inputValue == value);
					break;
				case DIFFERENT_TO:
					result = (inputValue != value);
					break;
				case LESS_THAN:
					result = (inputValue < value);
					break;
				case LESS_THAN_OR_EQUAL_TO:
					result = (inputValue <= value);
					break;
				case GREATER_THAN:
					result = (inputValue > value);
					break;
				case GREATER_THAN_OR_EQUAL_TO:
					result = (inputValue >= value);
					break;
				default:
					result = false;
			}
			output[id] = result ? 1 : 0;
		}
	}


	__global__ void UniformNormalDistribution(float *from, float *to, int size)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		float tmp;

		if (id < size)
		{
			tmp = normcdf(from[id] * sqrt((float)size));

			to[id] = (tmp -0.5)*2;
		}
	}

}
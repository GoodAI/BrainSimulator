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
	enum ActivationFunctionEnum
	{
		NO_ACTIVATION_FUNCTION,
		SIGMOID,
		IDENTITY,
		GAUSSIAN,
		RATIONAL_SIGMOID,
		RELU,
		SOFTMAX,
		TANH,
		LECUN_TANH,
	};

	__device__ int sign(float val)
	{
		return (val > 0) - (val < 0);
	}

	/* Transfer function definitions */
	__device__ float sigmoid(float x)
	{
		return 1.0f / (1.0f + expf(-x));
	}
	__device__ float sigmoid_derivative(float x)
	{
		return sigmoid(x) * (1.0f - sigmoid(x));
	}

	//__device__ float tanhf(float x) // already defined

	__device__ float tanh_derivative(float x)
	{
		float val = tanhf(x);
		return 1 - val * val;
	}

	// This activation function is recommended in 
	// "Efficient backprop" (LeCun 1998) and also discussed here:
	// http://stats.stackexchange.com/questions/38225/neural-net-cost-function-for-hyperbolic-tangent-activation
	__device__ float lecun_tanh(float x)
	{
		return 1.7159f * tanh((2.0f * x) / 3.0f);
	}

	__device__ float hyperbolic_secant(float x)
	{
		return 2.0f / (expf(x) + expf(-x));
	}

	__device__ float lecun_tanh_derivative(float x)
	{
		return 1.14393 * powf(hyperbolic_secant((2.0f * x) / 3.0f), 2.0f);
	}

	__device__ float identity(float x)
	{
		return x;
	}
	__device__ float identity_derivative(float x)
	{
		return 1.0f;
	}

	__device__ float gaussian(float x)
	{
		return expf(-pow(x, 2));
	}
	__device__ float gaussian_derivative(float x)
	{
		return -2.0f * x * gaussian(x);
	}

	__device__ float rationalsigmoid(float x)
	{
		return x / (1.0f + sqrt(1.0f + x * x));
	}
	__device__ float rationalsigmoid_derivative(float x)
	{
		float val = sqrt(1.0f + x * x);
		return 1.0f / (val * (1.0f + val));
	}

	__device__ float rectifiedLinearUnit(float x)
	{
		return (x > 0.0f) * x;
	}

	__device__ float rectifiedLinearUnit_derivative(float x)
	{
		return (x > 0.0f);
	}

	// not used for now
	__device__ float softmax(float* x, int index, int length)
	{
		float result = exp(x[index]);
		for (size_t i = 0; i < length; i++)
		{
			result /= exp(x[i]);
		}
		return result;
	}

	__device__ float softmax_derivative(float x)
	{
		return (x * (1 - x));
	}

	__device__ float RBMbinary(float x, float random)
	{
		return (random < sigmoid(x));
	}


	__device__ float Evaluate(ActivationFunctionEnum activationFunction, float input)
	{
		switch (activationFunction)
		{
			case SIGMOID:
				return sigmoid(input);

			case IDENTITY:
				return identity(input);

			case GAUSSIAN:
				return gaussian(input);

			case RATIONAL_SIGMOID:
				return rationalsigmoid(input);

			case RELU:
				return rectifiedLinearUnit(input);

			case SOFTMAX:
				// softmax is computed by a separate kernel (needs data from the whole layer)
				// to avoid infinity problems, handle softmax activation differently (use log trick)
				return input;

			case TANH:
				return tanhf(input);

			case NO_ACTIVATION_FUNCTION:
				return input;

			case LECUN_TANH:
				return lecun_tanh(input);

			default:
				return 0.0f;
		}
	}

	__device__ float EvaluateDerivative(ActivationFunctionEnum activationFunction, float input)
	{
		switch (activationFunction)
		{
			case SIGMOID:
				return sigmoid_derivative(input);

			case IDENTITY:
				return identity_derivative(input);

			case GAUSSIAN:
				return gaussian_derivative(input);

			case RATIONAL_SIGMOID:
				return rationalsigmoid_derivative(input);

			case RELU:
				return rectifiedLinearUnit_derivative(input);

			case SOFTMAX:
				return softmax_derivative(input);

			case TANH:
				return tanh_derivative(input);

			case NO_ACTIVATION_FUNCTION:
				return 1.0f;

			case LECUN_TANH:
				return lecun_tanh_derivative(input);

			default:
				return 0.0f;
		}
	}

	__global__ void SoftmaxKernel(
		float *outputPtr,
		float expSum,
		int layerSize
		)
	{
		// i: neuron id
		int i = blockDim.x * blockIdx.y * gridDim.x	//rows preceeding current row in grid
			+ blockDim.x * blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (i < layerSize)
		{
			// exp value is already present in the output array, so just divide by sum of exps (computed before kernel call)
			outputPtr[i] /= expSum;
		}


	}

}
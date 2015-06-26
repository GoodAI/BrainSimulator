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
		TANH
	};

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
		if (x > 0.0f)
			return x;
		return 0.0f;
	}

	__device__ float rectifiedLinearUnit_derivative(float x)
	{
		return x > 0.0f;
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

			case TANH:
				return tanhf(input);

			case NO_ACTIVATION_FUNCTION:
				return input;

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

			case TANH:
				return tanh_derivative(input);

			case NO_ACTIVATION_FUNCTION:
				return 1.0f;

			default:
				return 0.0f;
		}
	}
}
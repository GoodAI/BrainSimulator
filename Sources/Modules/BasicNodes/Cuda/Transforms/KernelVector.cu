#define _SIZE_T_DEFINED 

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>

extern "C"
{

	//  DEVICE KERNELS
	__forceinline__ __device__ int GetId()
	{
		return blockDim.x * blockIdx.y * gridDim.x //rows preceeding current row in grid
			+ blockDim.x * blockIdx.x             //blocks preceeding current block
			+ threadIdx.x;
	}

	// combine two vectors elemetwise with given weight, out = a * weightA + b * weightB
	__device__ void device_ElementwiseAdd_Weighted(
		float* a,
		float* b,
		float* out,
		float weightA,
		float weightB,
		int count
		)
	{
		int id = GetId();

		if (id < count)
		{
			out[id] = a[id] * weightA + b[id] * weightB;
		}
	}

	// GLOBAL KERNELS


	//Add scalar to each element of the input tensor
	__global__ void ScalarAdd(
		float* input,
		float scalar,
		float* output,
		int count
		)
	{
		int id = GetId();

		if (id < count)
		{
			output[id] = input[id] + scalar;
		}
	}

	//Add different scalar to each segment (elementwise)
	__global__ void ScalarAdd_Segmented(
		float* scalars,	//vector of N scalars
		float* input,	//input vector divisible into N segments
		float* output,	//output vector of the same length as input
		int noSegments, //number of segments == N
		int count		//length of the input vector
		)
	{
		int id = GetId();

		int segmentID;

		if (id < count)
		{
			segmentID = count / noSegments;
			output[id] = input[id] + scalars[segmentID];
		}
	}

	// O_i = scalar * B_i
	__global__ void ScalarMult(
		float * output,
		float * input,
		float scalar,
		int count
		)
	{
		int id = GetId();

		if (id < count)
		{
			output[id] = input[id] * scalar;
		}
	}

	// O_i = scalar1 * B_i + scalar2
	__global__ void ScalarMultThenAdd(
		float * output,
		float * input,
		float scalar1,
		float scalar2,
		int count
		)
	{
		int id = GetId();

		if (id < count)
		{
			output[id] = input[id] * scalar1 + scalar2;
		}
	}

	__global__ void ElementwiseAbs(
		float* a,
		float* result,
		int count
		)
	{
		int id = GetId();

		if (id < count)
		{
			if (a[id] < 0)
			{
				result[id] = -a[id];
			}
			else
			{
				result[id] = a[id];
			}
		}
	}

	__global__ void ElementwiseAdd(
		float* a,
		float* b,
		float* result,
		int count
		)
	{
		int id = GetId();

		if (id < count)
		{
			result[id] = a[id] + b[id];
		}
	}

	///Adds two vectores elementwise. bounding each element of the result
	__global__ void ElementwiseAdd_Bounded(
		float* a,
		float* b,
		float minBound,
		float maxBound,
		float* result,
		int count
		)
	{
		int id = GetId();

		if (id < count)
		{
			result[id] = a[id] + b[id];
			if (result[id] < minBound) {
				result[id] = minBound;
			}
			if (result[id] > maxBound) {
				result[id] = maxBound;
			}
		}
	}

	///Adds two vectores elementwise result = (1-bScale)*a+bScale*b. bounding each element of the result
	__global__ void ElementwiseAdd_BoundedWeighted(
		float* a,
		float* b,
		float bScale,
		float minBound,
		float maxBound,
		float* result,
		int count
		)
	{
		int id = GetId();

		if (id < count)
		{
			result[id] = (1 - bScale) * a[id] + bScale * b[id];
			if (result[id] < minBound) {
				result[id] = minBound;
			}
			if (result[id] > maxBound) {
				result[id] = maxBound;
			}
		}
	}

	__global__ void ElementwiseAdd_Weighted(
		float* a,
		float* b,
		float* out,
		float weightA,
		float weightB,
		int count
		)
	{
		device_ElementwiseAdd_Weighted(a, b, out, weightA, weightB, count);
	}

	// output = a./b
	__global__ void ElementwiseDiv(
		float* a,
		float* b,
		float* output,
		int count
		)
	{
		int id = GetId();

		if (id < count)
		{
			output[id] = a[id] / b[id]; //will be NaN if b[id] == 0, which is ok, at least we will notice in observers
		}
	}

	// elementwise multiplication of probabilities
	__global__ void ElementwiseMult(
		float* a,
		float* b,
		float* out,
		int count
		)
	{
		int id = GetId();

		if (id < count)
		{
			out[id] = a[id] * b[id];
		}
	}

	// elementwise multiplication of probabilities, first vector is (epected to be shorter) used multiple times over and over. Vectors b and out should have the same length.
	__global__ void ElementwiseMult_Segmented(
		float* a,
		float* b,
		float* out,
		int lengthA,
		int count
		)
	{
		int id = GetId();

		int idA;

		if (id < count)
		{
			idA = id % lengthA;
			out[id] = a[idA] * b[id];
		}
	}

	__global__ void ElementwiseSub(
		float* a,
		float* b,
		float* result,
		int count
		)
	{
		int id = GetId();

		if (id < count)
		{
			result[id] = a[id] - b[id];
		}
	}

	//multiplies two vectors as matrices so that result is a matrix where output_ij = a_i * b_j
	__global__ void CrossMult(
		float* a,
		float* b,
		float* output,	//matrix m * n, where m = length a and m = length b
		int lengthA,
		int lengthB
		)
	{
		int id = GetId();

		int i;
		int j;

		if (id < lengthA * lengthB)
		{
			i = id / lengthB;
			j = id % lengthB;

			output[id] = a[i] * b[j];
		}
	}

	__global__ void OtherAverage(
		float* a,
		float* b,
		float* result,
		int count
		)
	{
		device_ElementwiseAdd_Weighted(a, b, result, 0.5f, 0.5f, count);
	}

}


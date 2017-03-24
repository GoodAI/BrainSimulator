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

	__global__ void ElementwiseThreshold(
		float* output,
		float* input,
		float threshold,
		int count
	)
	{
		int id = GetId();

		if (id < count)
		{
			if (input[id] < threshold)
			{
				output[id] = threshold;
			}
			else
			{
				output[id] = input[id];
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

	///Adds two vectores elementwise result = weightA*a+weightB*b. bounding each element of the result
	__global__ void ElementwiseAdd_BoundedWeighted(
		float* result,
		float* a,
		float* b,
		float weightA,
		float weightB,
		float minBound,
		float maxBound,
		int count
	)
	{
		int id = GetId();

		if (id < count)
		{
			result[id] = weightA * a[id] + weightB * b[id];
			if (result[id] < minBound) {
				result[id] = minBound;
			}
			if (result[id] > maxBound) {
				result[id] = maxBound;
			}
		}
	}

	__global__ void ElementwiseAdd_Offsetted(
		float* result,
		float* a,
		float* b,
		int resultOffset,	// offset in the result
		int aOffset,		// offset in a indexes (0 means no offset)
		int bOffset,		// offset in b indexses
		int countSubtracted	// number of values to be added
	)
	{
		int id = GetId();

		if (id < countSubtracted)
		{
			result[id + resultOffset] = a[id + aOffset] + b[id + bOffset];
		}
	}

	// elementwise adition of two vectors, first vector is (expected to be shorter) used multiple times over and over. Vectors b and out should have the same length.
	__global__ void ElementwiseAdd_Segmented_Repeat(
		float* out,
		float* a,
		float* b,
		int lengthA,
		int count
	)
	{
		int id = GetId();

		int idA;

		if (id < count)
		{
			idA = id % lengthA;
			out[id] = a[idA] + b[id];
		}
	}

	// elementwise addition of two vectors, each element of the vector a (it is expected to be shorter than b) is used multiple times over whole segment of the vector b. Vectors b and out should have the same length.
	__global__ void ElementwiseAdd_Segmented_Stretch(
		float* out,
		float* a,
		float* b,
		int noSegments,  // = length A 
		int count  //length of the output and vector b
	)
	{
		int id = GetId();

		int segmentId;

		if (id < count)
		{
			segmentId = id / (count / noSegments);
			out[id] = a[segmentId] + b[id];
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

	// output is wa*a + wb*b, zero otherwise. a and b can be offsetted
	__global__ void ElementwiseAdd_WeightedOffsetted(
		float* a,
		float* b,
		float* out,
		float weightA,
		float weightB,
		int offsetA,
		int offsetB,
		int countA,
		int countB,
		int outputCount
	)
	{
		int id = GetId();

		if (id < outputCount)
		{
			out[id] = 0.0f;

			if (id >= offsetA && id < offsetA + countA) {
				out[id] += weightA * a[id - offsetA];
			}
			if (id >= offsetB && id < offsetB + countB) {
				out[id] += weightB * b[id - offsetB];
			}
		}
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


	// elementwise multiplication of two vectors, first vector is (expected to be shorter) used multiple times over and over. Vectors b and out should have the same length.
	__global__ void ElementwiseMult_Segmented_Repeat(
		float* out,
		float* a,
		float* b,
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

	// elementwise multiplication of two vectors, each element of the vector a (it is expected to be shorter than b) is used multiple times over whole segment of the vector b. Vectors b and out should have the same length.
	__global__ void ElementwiseMult_Segmented_Stretch(
		float* out,
		float* a,
		float* b,
		int lengthA,  // = noSegments 
		int count  //length of the output and vector b
	)
	{
		int id = GetId();

		int segmentId;

		if (id < count)
		{
			segmentId = id / (count / lengthA);
			out[id] = a[segmentId] * b[id];
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

	//multiplies each row from matrixA with each row from matrixB in a crossproduct manner, i.e., the two vectors are multiplied as matrices so that result is a matrix where output_ij = a_i * b_j. The overall result is then a tensor output_ijk, where k goes over rows in the matrices.
	__global__ void CrossMult_Segmented(
		float* output,	//tensor noColumnsA * noColumnsB * noRows
		float* matrixA,
		float* matrixB,
		int noColumnsA,
		int noColumnsB,
		int noRows
	)
	{
		int id = GetId();

		int i, j, k;

		if (id < noColumnsA * noColumnsB * noRows)
		{
			i = id % noColumnsA;                //columns in A
			j = (id / noColumnsA) % noColumnsB; //columns in B
			k = id / (noColumnsA * noColumnsB); //rows in both A and B, third dimension in the resulting tensor

			output[id] = matrixA[i + k * noColumnsA] * matrixB[j + k * noColumnsB];
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


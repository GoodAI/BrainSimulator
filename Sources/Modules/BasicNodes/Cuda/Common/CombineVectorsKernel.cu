#define _SIZE_T_DEFINED
#ifndef __cplusplus 
#define __cplusplus 
#endif

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <math_functions.h>
#include <float.h>
#include <cuComplex.h>

#include "Reduction\f_dot_f.cuh"
#include "Reduction\Reduction.cu"

#define ADD 0
#define SUB 1
#define MUL 2
#define AND 3
#define OR 4
#define OR_THRESHOLD 5
#define XOR 6
#define XNOR 7
#define IMP 8
#define PERM 9
#define INV_PERM 10
#define MODULO 11
#define DIVISION_INT 12
#define EQUAL 13


#define MAX_OPERANDS 20
#define MAX_SYMBOL_SIZE 4096

extern "C"
{
	//kernel code performs no binarity checks
	__global__ void CombineVectorsKernel(float** inputs, int inputsCount, float* output, int method, int count)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;


		if(threadId >= count)
			return;


		float out = inputs[0][threadId];

		switch (method)
		{
		case SUB:
			for (int i = 1; i < inputsCount; i++)
				out -= inputs[i][threadId];
			break;

		case ADD:
			for (int i = 1; i < inputsCount; i++)
				out += inputs[i][threadId];
			break;

		case AND:
		case MUL:
			for (int i = 1; i < inputsCount; i++)
				out *= inputs[i][threadId];
			break;

		case OR:
			for (int i = 1; i < inputsCount; i++)
				out += inputs[i][threadId];

			out = out >= 1;
			break;

		case OR_THRESHOLD:
			for (int i = 1; i < inputsCount; i++)
				out += inputs[i][threadId];

			out = out >= (inputsCount * 0.5f);
			break;

		case XOR:
			for (int i = 1; i < inputsCount; i++)
				out += inputs[i][threadId];

			out = ((int)out) % 2;
			break;

		case XNOR:
			for (int i = 1; i < inputsCount; i++)
				out += inputs[i][threadId];

			out = ((int)out + 1) % 2;
			break;

		case PERM:
			__shared__ float tmp[MAX_SYMBOL_SIZE];

			tmp[threadId] = out;

			__threadfence();

			for (int i = 1; i < inputsCount; i++)
			{
				float val = tmp[__float2int_rn(inputs[i][threadId])];
				__syncthreads();

				tmp[threadId] = val;
				__threadfence();
			}

			out = tmp[threadId];
			break;

		case INV_PERM:
			__shared__ float i_tmp[MAX_SYMBOL_SIZE];

			i_tmp[threadId] = out;

			__threadfence();

			for (int i = 1; i < inputsCount; i++)
			{
				int idx = __float2int_rn(inputs[i][threadId]);
				float val = i_tmp[threadId];

				__syncthreads();

				i_tmp[idx] = val;
				__threadfence();
			}

			out = i_tmp[threadId];
			break;

		case EQUAL: // Warning: uses a strict equality comparison on floats
		{
			bool eq = true;
			for (int i = 1; eq && (i < inputsCount); i++)
			{
				eq = (eq && (out == inputs[i][threadId]));
			}
			out = eq ? 1.0f : 0.0f;
			break;
		}
		default:
			break;
		}

		output[threadId] = out;
	}


	__device__ __forceinline__ void CombineTwoVectorsInternal(const float& input1, const float& input2, float& output, int method)
	{
		switch (method)
		{
		case SUB:
		{
			output = input1 - input2;
			break;
		}

		case ADD:
		{
			output = input1 + input2;
			break;
		}
		case AND:
		case MUL:
		{
			output = input1 * input2;
			break;
		}
		case OR:
		case OR_THRESHOLD:
		{
			output = (input1 + input2) >= 1;
			break;
		}
		case XOR:
		{
			output = (input1 + input2) == 1;
			break;
		}
		case XNOR:
		{
			output = (input1 + input2) != 1;
			break;
		}
		case IMP:
		{
			output = input1 <= input2;
			break;
		}
		case MODULO:
		{
			int mod = __float2int_rn(input2);
			int n = __float2int_rd(input1 / mod);
			output = input1 - mod * n;
			break;
		}
		case DIVISION_INT:
		{
			output = __float2int_rz(input1 / input2);
			break;
		}
		case EQUAL:
		{
			output = (input1 == input2) ? 1.0f : 0.0f;
			break;
		}
		default:
			break;
		}	
	}

	__global__ void CombineTwoVectorsKernel(const float* input1, const float* input2, float* output, int method, int count)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (threadId >= count)
			return;


		switch (method)
		{
		case PERM:
			{
				float tmp = input1[(int)input2[threadId]];

				if (input1 == output)
					__threadfence();

				output[threadId] = tmp;
				break;
			}

		case INV_PERM:
			{
				int idx = (int)input2[threadId];

				if (input1 == output)
					__threadfence();

				output[idx] = input1[threadId];
				break;
			}

		default:
			CombineTwoVectorsInternal(input1[threadId], input2[threadId], output[threadId], method);
			break;
		}
	}

	__device__ __forceinline__ void CombineTwoVectorsKernelVarSizeInternal(const float* input1, const float* input2, float* output, int method, int count1, int count2)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;


		switch (method)
		{
		case PERM:
			{
				if (count2 > count1)
					return;

				float tmp = input1[(int)input2[threadId]];

				if (input1 == output)
					__threadfence();

				output[threadId] = tmp;
				break;
			}
		case INV_PERM:
			{
				if (count2 > count1)
					return;

				int idx = (int)input2[threadId];

				if (input1 == output)
					__threadfence();

				output[idx] = input1[threadId];
				break;
			}

		default:
			{
				int minCount = count1 <= count2 ? count1 : count2;

				if (threadId < minCount)
				{
					CombineTwoVectorsInternal(input1[threadId], input2[threadId], output[threadId], method);
					return;
				}


				if (count1 > count2)
				{
					if (threadId < count1)
						output[threadId] = input1[threadId];
				}
				else if (count2 > count1)
				{
					if (threadId < count2)
						output[threadId] = method == SUB ? -input2[threadId] : input2[threadId];
				}

				break;
			}
		}
	}

	__global__  void CombineTwoVectorsKernelVarSize(float* input1, float* input2, float* output, int method, int count1, int count2)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;


		if (count1 > 1)
		{
			if (count2 > 1)
			{
				CombineTwoVectorsKernelVarSizeInternal(input1, input2, output, method, count1, count2);
			}
			else if (threadId < count1)
			{
				CombineTwoVectorsInternal(input1[threadId], input2[0], output[threadId], method);
			}
		}
		else
		{
			if (count2 > 1 && threadId < count2)
			{
				CombineTwoVectorsInternal(input1[0], input2[threadId], output[threadId], method);
			}
			else
			{
				CombineTwoVectorsInternal(input1[0], input2[0], output[threadId], method);
			}
		}
	}


	__global__ void AddToIdcs(float* source, const float* idcs, float* target, int method, int idcsCount)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (threadId >= idcsCount) // Should be true: idcsCount == sourceCount
			return;


		float& tar = target[__float2int_rn(idcs[threadId])];
		float& src = source[threadId];

		switch (method)
		{
		case ADD:
			atomicAdd(&tar, src);
			break;

		case SUB:
			atomicAdd(&tar, -src);
			break;

		case OR:
			tar = src;
			break;

		default:
			break;
		}
	}

	__global__ void MapToIdcs(float* source, float* sourceLengthSq, const float* idcs, float* target, int idcsCount)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (threadId >= idcsCount) // Should be true: idcsCount == sourceCount
			return;


		float& tar = target[__float2int_rn(idcs[threadId])];
		float& src = source[threadId];

		float len = *sourceLengthSq;

		if (len < 0.0000001f)
			return;

		len = 1 / sqrtf(len);

		// Write the normalized vector back to output
		CombineTwoVectorsInternal(src, len, tar, MUL);
	}


	__global__ void LengthFromElements(float* element1, float* element2, float* output, int count)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if(threadId < count)
		{
			output[threadId] = sqrtf(element1[threadId] * element1[threadId] + element2[threadId] * element2[threadId]);
		}
	}

	__global__ void MulComplexElementWise(cuFloatComplex* input1, cuFloatComplex* input2, cuFloatComplex* output, int count)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if(threadId < count)
		{			
			cuFloatComplex i1 = input1[threadId];
			cuFloatComplex i2 = input2[threadId];

			output[threadId] = cuCmulf(i1, i2);								
		}
	}

	__global__ void InvolveVector(float* input, float* output, int inputSize)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if(threadId < inputSize - 1)
		{									
			output[0] = input[0];
			output[threadId + 1] = input[inputSize - threadId - 1];
		}
	}

	__global__ void Interpolate(float* input1, float* input2, float* output, float weight, int inputSize)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if(threadId < inputSize)
		{												
			if (weight <= 0)
			{
				output[threadId] = input1[threadId];
			}
			else if (weight >= 1)
			{
				output[threadId] = input2[threadId];
			}
			else
			{
				output[threadId] = (1 - weight) * input1[threadId] + weight * input2[threadId];
			}
		}
	}

	__global__ void InterpolateFromMemBlock(float* input1, float* input2, float* output, float* weightMemBlock, int inputSize)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if(threadId < inputSize)
		{					
			if (weightMemBlock[0] <= 0)
			{
				output[threadId] = input1[threadId];
			}
			else if (weightMemBlock[0] >= 1)
			{
				output[threadId] = input2[threadId];
			}
			else
			{
				output[threadId] = (1 - weightMemBlock[0]) * input1[threadId] + weightMemBlock[0] * input2[threadId];
			}
		}
	}


	// naive mat. multiplication
	// TODO: rewrite it with sync_threads... :)   Check out nvida dev-blog or TestFeat/HMath.cu how it will be...
	__global__ void MatMultipl_naive (float * A, float * B, float * C , int nColsA , int nColsB , int sizeC ) {
		int i_col = blockIdx.x * blockDim.x + threadIdx.x; /// index in row
		int i_row = blockIdx.y * blockDim.y + threadIdx.y; /// index in column
		int idx = i_row * nColsB + i_col;  // # of cols in B = # of cols in C
		float Cvalue = 0;

		if (idx < sizeC){
			for (int e=0; e < nColsA; e++)
				Cvalue += A[i_row * nColsA + e] * B[e * nColsB + i_col];
			C[idx] = Cvalue;
		}
	}
}

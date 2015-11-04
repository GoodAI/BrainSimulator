#define _SIZE_T_DEFINED 

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>

#include "../Observers/ColorScaleObserverSingle.cu"

#include "MurmurHash3.cu"


extern "C"
{
	__global__ void Transpose(float* input, float* output, int rows, int cols)
	{
		int tid = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x
			+ threadIdx.x;

		int x = tid / rows;
		int y = tid % rows;

		if (tid >= rows * cols)
			return;

		float temp = input[y * cols + x];

		if (input == output)
			__syncthreads();

		output[x * rows + y] = temp;
	}


	// KeySize should be equel to SymbolSize for now
	// BinSize should be a power of two for now
	// Using the Murmurhash3 hash function: https://code.google.com/p/smhasher/wiki/MurmurHash3
	__device__ __forceinline__ void GetIndicesInternal(const unsigned int* keys, float* output, int keySize, int outputSize, int binSize, int seed)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		if (threadId >= outputSize)
			return;

		int temp = keys[threadId];

		MurmurHash3_x86_32(&temp, sizeof(int), seed, &temp);

		output[threadId] = temp & (binSize - 1); // == key % binSize
	}

	__global__ void GetIndices(const unsigned int* keys, float* output, int keySize, int outputSize, int binSize, int seed)
	{
		GetIndicesInternal(keys, output, keySize, outputSize, binSize, seed);
	}

	__global__ void GetIndices_ImplicitSeed(const unsigned int* keys, float* output, int keySize, int outputSize, int binSize, int seed)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x				//blocks preceeding current block
			+ threadIdx.x;

		int temp = threadId;

		MurmurHash3_x86_32(&temp, sizeof(int), seed, &temp);
		GetIndicesInternal(keys, output, keySize, outputSize, binSize, temp ^ seed);
	}

	__global__ void GetIndices_NoHashing(float* keys, float* output, int keySize, int internalBinCount)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
			+ blockDim.x*blockIdx.x						//blocks preceeding current block
			+ threadIdx.x;

		int temp = threadId;

		if (threadId < keySize){
			output[threadId] = internalBinCount*threadId + keys[threadId];
		}
	}
}

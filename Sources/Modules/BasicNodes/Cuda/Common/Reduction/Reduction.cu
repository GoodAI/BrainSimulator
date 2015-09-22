#include <cuda.h>
#include <iostream>
#include <iomanip>

#include "SharedMemory.cuh"

// INTEGER BASED
#include "i_Sum_i.cuh"
#include "i_MinIdx_2i.cuh"
#include "i_MaxIdx_2i.cuh"
#include "i_MinIdxMaxIdx_4i.cuh"

// SINGLE BASED
#include "f_Sum_f.cuh"
#include "f_MinMax_2f.cuh"
#include "f_MinIdx_fi.cuh"
#include "f_MinIdx_ff.cuh"
#include "f_MaxIdx_fi.cuh"
#include "f_MaxIdx_ff.cuh"
#include "f_MinMax_2f.cuh"
#include "f_MinIdxMaxIdx_fifi.cuh"

// DOT PRODUCT BASED
#include "i_Dot_i.cuh"
#include "f_Dot_f.cuh"
#include "f_Cosine_f.cuh"

using namespace std;

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError(cudaError_t err, const char *file, int line)
{
	if(err != cudaSuccess)
	{
		printf("%s in %s at line %d \n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}

int randl()
{
	return (rand()<<16) + rand();
}

template<typename R, unsigned int tCnt, bool finalize>
__forceinline__ __device__ void LogStepShared(R* out, volatile R* partials)
{
	const unsigned int tid = threadIdx.x;
	if (tCnt >= 1024)
	{
		if (tid < 512) { 
			partials[tid].op(partials[tid + 512]);
		}
		__syncthreads();
	}
	if (tCnt >= 512)
	{
		if (tid < 256) { 
			partials[tid].op(partials[tid + 256]);
		}
		__syncthreads();
	}
	if (tCnt >= 256) {
		if (tid < 128) {
			partials[tid].op(partials[tid + 128]);
		}
		__syncthreads();
	}
	if (tCnt >= 128) {
		if (tid < 64) {
			partials[tid].op(partials[tid + 64]);
		}
		__syncthreads();
	}
	
	if (tid < 32)
	{
		if (tCnt >= 64) { partials[tid].op(partials[tid + 32]); }
		if (tCnt >= 32) { partials[tid].op(partials[tid + 16]); }
		if (tCnt >= 16) { partials[tid].op(partials[tid + 8]); }
		if (tCnt >= 8) { partials[tid].op(partials[tid + 4]); }
		if (tCnt >= 4) { partials[tid].op(partials[tid + 2]); }
		if (tCnt >= 2) { partials[tid].op(partials[tid + 1]); }
		
		if (tid == 0)
		{
			if (finalize) partials[0].finalize(out);
			else *out = partials[0];
		}
	}
}

__device__ int buffer[8192];
__device__ unsigned int barrier = 0;

template<typename R, typename T, unsigned int tCnt>
__forceinline__ __device__ void DReduction(void* rawOut, volatile const void* rawIn, unsigned int size, unsigned int outOff, unsigned int inOff, unsigned int stride, bool distributed)
{
	__syncthreads();

	R* out = reinterpret_cast<R*>(buffer);
	volatile const T* in = reinterpret_cast<volatile const T*>(rawIn) + inOff;

	SharedMemory<R> sPartials;
	const unsigned int tid = threadIdx.x;

	unsigned int gridDim_x = distributed ? 1 : gridDim.x;
	unsigned int blockIdx_x = distributed ? 0 : blockIdx.x;

	R sum;
	for (unsigned int i = stride * (blockIdx_x * tCnt + tid); i < size; i += stride * tCnt * gridDim_x)
	{
		sum.op(in[i], i + inOff);
	}
	sPartials[tid] = sum;
	__syncthreads();

	if (gridDim_x == 1)
	{
		out = reinterpret_cast<R*>(reinterpret_cast<char*>(rawOut)+R::outSize*outOff);
		LogStepShared<R,tCnt,false>(out, sPartials);
		return;
	}
	LogStepShared<R,tCnt,false>(&out[blockIdx_x], sPartials);

	__shared__ bool lastBlock;
	__threadfence();

	if (tid == 0)
	{
		unsigned int ticket = atomicAdd(&barrier, 1);
		lastBlock = (ticket == gridDim_x - 1);
	}
	__syncthreads();

	if (lastBlock)
	{
		R sum;
		for (unsigned int i = tid; i < gridDim_x; i += tCnt)
		{
			sum.op(out[i]);
		}
		sPartials[threadIdx.x] = sum;
		__syncthreads();

		out = reinterpret_cast<R*>(reinterpret_cast<char*>(rawOut)+R::outSize*outOff);
		LogStepShared<R,tCnt,false>(out, sPartials);
		barrier = 0;
	}
}

template<typename R, typename T, unsigned int tCnt>
__global__ void Reduction(void* rawOut, volatile const void* rawIn, unsigned int size, unsigned int outOff, unsigned int inOff, unsigned int stride, bool distributed)
{
	DReduction<R,T,tCnt>(rawOut, rawIn, size, outOff, inOff, stride, distributed);
}

template<typename R, typename T, unsigned int tCnt>
__forceinline__ __device__ void DDotProduct(void* rawOut, unsigned int outOff, volatile const void* rawIn1, volatile const void* rawIn2, unsigned int size, bool distributed)
{
	__syncthreads();
	
	R* out = reinterpret_cast<R*>(buffer);
	volatile const T* in1 = reinterpret_cast<volatile const T*>(rawIn1);
	volatile const T* in2 = reinterpret_cast<volatile const T*>(rawIn2);

	SharedMemory<R> sPartials;
	const unsigned int tid = threadIdx.x;

	unsigned int gridDim_x = distributed ? 1 : gridDim.x;
	unsigned int blockIdx_x = distributed ? 0 : blockIdx.x;

	R sum;
	for (unsigned int i = blockIdx_x * tCnt + tid; i < size; i += tCnt * gridDim_x)
	{
		sum.op(in1[i], in2[i], i);
	}
	sPartials[tid] = sum;
	__syncthreads();

	if (gridDim_x == 1)
	{
		out = reinterpret_cast<R*>(reinterpret_cast<char*>(rawOut)+R::outSize*outOff);
		LogStepShared<R,tCnt,true>(out, sPartials);
		return;
	}
	LogStepShared<R,tCnt,false>(&out[blockIdx_x], sPartials);

	__shared__ bool lastBlock;
	__threadfence();

	if (tid == 0)
	{
		unsigned int ticket = atomicAdd(&barrier, 1);
		lastBlock = (ticket == gridDim_x - 1);
	}
	__syncthreads();

	if (lastBlock)
	{
		R sum;
		for (unsigned int i = tid; i < gridDim_x; i += tCnt)
		{
			sum.op(out[i]);
		}
		sPartials[threadIdx.x] = sum;
		__syncthreads();

		out = reinterpret_cast<R*>(reinterpret_cast<char*>(rawOut)+R::outSize*outOff);
		LogStepShared<R,tCnt,true>(out, sPartials);
		barrier = 0;
	}
}

template<typename R, typename T, unsigned int tCnt>
__global__ void DotProduct(void* rawOut, unsigned int outOff, volatile const void* rawIn1, volatile const void* rawIn2, unsigned int size, bool distributed)
{
	DDotProduct<R,T,tCnt>(rawOut, outOff, rawIn1, rawIn2, size, distributed);
}

template<typename R, typename T>
void ReductionTemplate()
{
	Reduction<R,T,1024><<<0,0>>>(0,0,0,0,0,0,0);
}

template<typename R, typename T>
void DotProductTemplate()
{
	DotProduct<R,T,1024><<<0,0>>>(0,0,0,0,0,0);
}

extern "C"
void InstantiationDummy()
{
	// INTEGER BASED
	ReductionTemplate < i_Sum_i, int > ();
	ReductionTemplate < i_MinIdx_2i, int > ();
	ReductionTemplate < i_MaxIdx_2i, int > ();
	ReductionTemplate < i_MinIdxMaxIdx_4i, int > ();

	// SINGLE BASED
	ReductionTemplate < f_Sum_f, float > ();
	ReductionTemplate < f_MinMax_2f, float > ();
	ReductionTemplate < f_MinIdx_fi, float > ();
	ReductionTemplate < f_MinIdx_ff, float > ();
	ReductionTemplate < f_MaxIdx_fi, float > ();
	ReductionTemplate < f_MaxIdx_ff, float > ();
	ReductionTemplate < f_MinIdxMaxIdx_fifi, float > ();

	// DOT PRODUCT
	DotProductTemplate <i_Dot_i, int > ();
	DotProductTemplate <f_Dot_f, float > ();
	DotProductTemplate <f_Cosine_f, float > ();
}

typedef void(*reduction)(void*, volatile const void*, unsigned int, unsigned int, unsigned int, unsigned int, bool);

template<typename R, typename T, const int bCnt>
void TestReduction(reduction kernel, const char* name, int repetitions, int sizeMax, int min, int max, float div, bool distributed)
{
	const int w = 20;
	for (int r = 0; r < repetitions; ++r)
	{
		cudaEvent_t startGPU, stopGPU;
		HANDLE_ERROR(cudaEventCreate(&startGPU));
		HANDLE_ERROR(cudaEventCreate(&stopGPU));
		float timeGPU;
		float timeCPU;

		int inSize = randl() % sizeMax + 1;
		int inOff = randl() % inSize;
		int size = randl() % (inSize - inOff) + 1;
		T* d_in, *h_in = new T[inSize];
		HANDLE_ERROR(cudaMalloc(&d_in, sizeof(T) * inSize));
		for (int i = 0; i < inSize; ++i)
		{
			h_in[i] = static_cast<T>(randl() % (max - min) + min) / div;
		}
		HANDLE_ERROR(cudaMemcpy(d_in, h_in, sizeof(T) * inSize, cudaMemcpyHostToDevice));

		int stride = 1;
		if (randl() % 2 == 0) stride = randl() % (32) + 1;

		int outOff = distributed ? bCnt : randl() % 1000;
		R* d_out;
		R* h_out = reinterpret_cast<R*>(new char[R::outSize*(outOff+1)]);
		R* c_out = reinterpret_cast<R*>(new char[R::outSize*(outOff+1)]);
		HANDLE_ERROR(cudaMalloc(&d_out, R::outSize * (outOff + 1)));
		HANDLE_ERROR(cudaMemcpy(d_out, h_out, R::outSize * (outOff + 1), cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaEventRecord(startGPU, 0));

		if (distributed)
			for (size_t b = 0; b < bCnt; b++)
			{
				unsigned int chunkSize = size / bCnt;
				unsigned int chunkOffset = b * chunkSize;
				kernel <<<bCnt,1024,sizeof(R)*1024>>>(d_out, d_in, chunkSize, b, inOff + chunkOffset, stride, distributed);
			}
		else kernel<<<bCnt,1024,sizeof(R)*1024>>>(d_out, d_in, size, outOff, inOff, stride, distributed);

		HANDLE_ERROR(cudaEventRecord(stopGPU, 0));
		HANDLE_ERROR(cudaEventSynchronize(stopGPU));
		HANDLE_ERROR(cudaEventElapsedTime(&timeGPU, startGPU, stopGPU));

		HANDLE_ERROR(cudaMemcpy(h_out, d_out, R::outSize * (outOff + 1), cudaMemcpyDeviceToHost));

		time_t startCPU, stopCPU;
		int cycles = 100000000 / (inSize - inOff) >= 1 ? 100000000 / (inSize - inOff) : 1;
		startCPU = clock();

		if (distributed)
			for (size_t b = 0; b < bCnt; ++b)
			{
				unsigned int chunkSize = size / bCnt;
				unsigned int chunkOffset = b * chunkSize;
				for (size_t c = 0; c < cycles; ++c)
					R::simulate(c_out, h_in, chunkSize, b, inOff + chunkOffset, stride);
			}
		else
			for (size_t c = 0; c < cycles; ++c)
				R::simulate(c_out, h_in, size, outOff, inOff, stride);


		stopCPU = clock();
		timeCPU = difftime(stopCPU, startCPU) / cycles;

		cout << "=== Test: " << name << " ===" << endl;
		cout << left << setw(w) << "Speedup" << setw(w) << "GPU time" << setw(w) << "CPU time" << setw(w)
			<< "input size" << setw(w) << "size" << setw(w) << "output offset" << setw(w) << "input offset" << setw(w) << "stride" << endl;
		cout << left << setw(w) << (timeCPU / timeGPU) << setw(w) << timeGPU << setw(w) << timeCPU << setw(w) << inSize << setw(w) << size
			<< setw(w) << outOff << setw(w) << inOff << setw(w) << stride << endl;

		printf("Check (GPU == CPU): \n");
		bool passed = true;
		if (distributed)
			for (size_t b = 0; b < bCnt; b++)
				passed &= R::check(h_out, c_out, b, h_in);
		else
			passed = R::check(h_out, c_out, outOff, h_in);

		printf("------------\n");
		if (passed) printf("|  PASSED  |\n");
		else printf("|  FAILED  |\n");
		printf("------------\n\n");

		HANDLE_ERROR(cudaFree(d_in));
		HANDLE_ERROR(cudaFree(d_out));

		delete[] h_in;
		delete[] h_out;
		delete[] c_out;
	}
}

typedef void(*dotproduct)(void*, unsigned int, volatile const void*, volatile const void*, unsigned int, bool);

template<typename R, typename T, const int bCnt>
void TestDotProduct(dotproduct kernel, const char* name, int repetitions, int sizeMax, int min, int max, float div, bool distributed)
{
	const int w = 20;
	for (int r = 0; r < repetitions; ++r)
	{
		cudaEvent_t startGPU, stopGPU;
		HANDLE_ERROR(cudaEventCreate(&startGPU));
		HANDLE_ERROR(cudaEventCreate(&stopGPU));
		float timeGPU;
		float timeCPU;

		int size = randl() % sizeMax + 1;
		T* d_in1, * d_in2, *h_in1 = new T[size], *h_in2 = new T[size];
		HANDLE_ERROR(cudaMalloc(&d_in1, sizeof(T) * size));
		HANDLE_ERROR(cudaMalloc(&d_in2, sizeof(T) * size));
		for (int i = 0; i < size; ++i)
		{
			h_in1[i] = static_cast<T>(float(randl() % (max - min) + min) / div);
			h_in2[i] = static_cast<T>(float(randl() % (max - min) + min) / div);
		}
		HANDLE_ERROR(cudaMemcpy(d_in1, h_in1, sizeof(T) * size, cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(d_in2, h_in2, sizeof(T) * size, cudaMemcpyHostToDevice));

		int outOff = distributed ? bCnt : randl() % 1000;
		R* d_out;
		R* h_out = reinterpret_cast<R*>(new char[R::outSize*(outOff+1)]);
		R* c_out = reinterpret_cast<R*>(new char[R::outSize*(outOff+1)]);
		HANDLE_ERROR(cudaMalloc(&d_out, R::outSize * (outOff + 1)));
		HANDLE_ERROR(cudaMemcpy(d_out, h_out, R::outSize * (outOff + 1), cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaEventRecord(startGPU, 0));

		if (distributed)
			for (size_t b = 0; b < bCnt; ++b)
			{
				unsigned int chunkSize = size / bCnt;
				unsigned int chunkOffset = b * chunkSize;
				kernel<<<bCnt,1024,sizeof(R)*1024>>>(d_out, b, d_in1 + chunkOffset, d_in2 + chunkOffset, chunkSize, distributed);
			}
		else
			kernel<<<bCnt,1024,sizeof(R)*1024>>>(d_out, outOff, d_in1, d_in2, size, distributed);

		HANDLE_ERROR(cudaEventRecord(stopGPU, 0));
		HANDLE_ERROR(cudaEventSynchronize(stopGPU));
		HANDLE_ERROR(cudaEventElapsedTime(&timeGPU, startGPU, stopGPU));

		HANDLE_ERROR(cudaMemcpy(h_out, d_out, R::outSize * (outOff + 1), cudaMemcpyDeviceToHost));

		time_t startCPU, stopCPU;
		int cycles = 100000000 / size >= 1 ? 100000000 / size : 1;
		startCPU = clock();

		if (distributed)
			for (size_t b = 0; b < bCnt; ++b)
			{
				unsigned int chunkSize = size / bCnt;
				unsigned int chunkOffset = b * chunkSize;
				for (int c = 0; c < cycles; ++c)
					R::simulate(c_out, b, h_in1 + chunkOffset, h_in2 + chunkOffset, chunkSize);
			}
		else
			for (size_t c = 0; c < cycles; ++c)
				R::simulate(c_out, outOff, h_in1, h_in2, size);

		stopCPU = clock();
		timeCPU = difftime(stopCPU, startCPU) / cycles;

		cout << "=== Test: " << name << " ===" << endl;
		cout << left << setw(w) << "Speedup" << setw(w) << "GPU time" << setw(w) << "CPU time" << setw(w) << "size"
			<< setw(w) << "output offset" << endl;
		cout << left << setw(w) << (timeCPU / timeGPU) << setw(w) << timeGPU << setw(w) << timeCPU << setw(w) << size
			<< setw(w) << outOff << endl;

		printf("Check (GPU == CPU): \n");
		bool passed = true;
		if (distributed)
			for (size_t b = 0; b < bCnt; b++)
				passed &= R::check(h_out, c_out, b, h_in1, h_in2);
		else
			passed = R::check(h_out, c_out, outOff, h_in1, h_in2);

		printf("------------\n");
		if (passed) printf("|  PASSED  |\n");
		else printf("|  FAILED  |\n");
		printf("------------\n\n");

		HANDLE_ERROR(cudaFree(d_in1));
		HANDLE_ERROR(cudaFree(d_in2));
		HANDLE_ERROR(cudaFree(d_out));

		delete[] h_in1;
		delete[] h_in2;
		delete[] h_out;
		delete[] c_out;
	}
}

int main(int argc, char* argv[])
{
	srand(time(NULL));

	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, 0);

	int repetitions = 10;
	int sizeMax = 10000000;

	// INTEGER BASED
	TestReduction<i_Sum_i, int, 10>(Reduction<i_Sum_i, int, 1024>, "Reduction i_Sum_i", repetitions, sizeMax, -10, 10, 1, false);
	TestReduction<i_MinIdx_2i, int, 10>(Reduction<i_MinIdx_2i, int, 1024>, "Reduction i_MinIdx_2i", repetitions, sizeMax, -10000, 10000, 1, false);
	TestReduction<i_MaxIdx_2i, int, 10>(Reduction<i_MaxIdx_2i, int, 1024>, "Reduction i_MaxIdx_2i", repetitions, sizeMax, -10000, 10000, 1, false);
	TestReduction<i_MinIdxMaxIdx_4i, int, 10>(Reduction<i_MinIdxMaxIdx_4i, int, 1024>, "Reduction i_MinIdxMaxIdx_4i", repetitions, sizeMax, 0, 10000, 1, false);

	// INTEGER BASED DISTRIBUTED
	TestReduction<i_Sum_i, int, 10>(Reduction<i_Sum_i, int, 1024>, "Reduction i_Sum_i", repetitions, sizeMax, -10, 10, 1, true);
	TestReduction<i_MinIdx_2i, int, 10>(Reduction<i_MinIdx_2i, int, 1024>, "Reduction i_MinIdx_2i", repetitions, sizeMax, -10000, 10000, 1, true);
	TestReduction<i_MaxIdx_2i, int, 10>(Reduction<i_MaxIdx_2i, int, 1024>, "Reduction i_MaxIdx_2i", repetitions, sizeMax, -10000, 10000, 1, true);
	TestReduction<i_MinIdxMaxIdx_4i, int, 10>(Reduction<i_MinIdxMaxIdx_4i, int, 1024>, "Reduction i_MinIdxMaxIdx_4i", repetitions, sizeMax, 0, 10000, 1, true);

	// SINGLE BASED
	TestReduction<f_Sum_f, float, 10>(Reduction<f_Sum_f, float, 1024>, "Reduction f_Sum_f", repetitions, sizeMax, -100, 100, 100, false);
	TestReduction<f_MinMax_2f, float, 10>(Reduction<f_MinMax_2f, float, 1024>, "Reduction f_MinMax_2f", repetitions, sizeMax, -100000, 100000, 1000, false);
	TestReduction<f_MinIdx_fi, float, 10>(Reduction<f_MinIdx_fi, float, 1024>, "Reduction f_MinIdx_fi", repetitions, sizeMax, -100000, 100000, 1000, false);
	TestReduction<f_MaxIdx_fi, float, 10>(Reduction<f_MaxIdx_fi, float, 1024>, "Reduction f_MaxIdx_fi", repetitions, sizeMax, -100000, 100000, 1000, false);
	TestReduction<f_MinIdxMaxIdx_fifi, float, 10>(Reduction<f_MinIdxMaxIdx_fifi, float, 1024>, "Reduction f_MinIdxMaxIdx_fifi", repetitions, sizeMax, 0, 100000, 1000, false);

	// SINGLE BASED DISTRIBUTED
	TestReduction<f_Sum_f, float, 10>(Reduction<f_Sum_f, float, 1024>, "Reduction f_Sum_f", repetitions, sizeMax, -100, 100, 100, true);
	TestReduction<f_MinMax_2f, float, 10>(Reduction<f_MinMax_2f, float, 1024>, "Reduction f_MinMax_2f", repetitions, sizeMax, -100000, 100000, 1000, true);
	TestReduction<f_MinIdx_fi, float, 10>(Reduction<f_MinIdx_fi, float, 1024>, "Reduction f_MinIdx_fi", repetitions, sizeMax, -100000, 100000, 1000, true);
	TestReduction<f_MaxIdx_fi, float, 10>(Reduction<f_MaxIdx_fi, float, 1024>, "Reduction f_MaxIdx_fi", repetitions, sizeMax, -100000, 100000, 1000, true);
	TestReduction<f_MinIdxMaxIdx_fifi, float, 10>(Reduction<f_MinIdxMaxIdx_fifi, float, 1024>, "Reduction f_MinIdxMaxIdx_fifi", repetitions, sizeMax, 0, 100000, 1000, true);

	// DOT PRODUCT
	TestDotProduct<i_Dot_i, int, 10>(DotProduct<i_Dot_i, int, 1024>, "DotProduct i_Dot_i", repetitions, sizeMax, -10, 10, 1, false);
	TestDotProduct<f_Dot_f, float, 10>(DotProduct<f_Dot_f, float, 1024>, "DotProduct f_Dot_f", repetitions, sizeMax, -100, 100, 100, false);
	TestDotProduct<f_Cosine_f, float, 10>(DotProduct<f_Cosine_f, float, 1024>, "DotProduct f_Cosine_f", repetitions, sizeMax, -100, 100, 100, false);

	// DOT PRODUCT DISTRIBUTED
	TestDotProduct<i_Dot_i, int, 10>(DotProduct<i_Dot_i, int, 1024>, "DotProduct i_Dot_i", repetitions, sizeMax, -10, 10, 1, true);
	TestDotProduct<f_Dot_f, float, 10>(DotProduct<f_Dot_f, float, 1024>, "DotProduct f_Dot_f", repetitions, sizeMax, -100, 100, 100, true);
	TestDotProduct<f_Cosine_f, float, 10>(DotProduct<f_Cosine_f, float, 1024>, "DotProduct f_Cosine_f", repetitions, sizeMax, -100, 100, 100, true);

	return 0;
}

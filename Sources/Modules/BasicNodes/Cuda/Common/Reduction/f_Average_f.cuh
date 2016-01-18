#pragma once

#include <cfloat>
#include <cmath>

#include "SharedMemory.cuh"

class f_Average_f
{
public:
	static const int outSize;

	float m_average;

	__forceinline__ __host__ __device__
	f_Average_f()
	{
		m_average = 0;
	}

	__forceinline__ __host__ __device__
	f_Average_f& op(float x, int idx)
	{
		m_average += x / (float)gridDim.x;
		return *this;
	}

	__forceinline__ __host__ __device__
	volatile f_Average_f& op(float x, int idx) volatile
	{
		m_average += x / (float)gridDim.x;
		return *this;
	}

	__forceinline__ __host__ __device__
	f_Average_f& op(const f_Average_f& x)
	{
		m_average += x.m_average;
		return *this;
	}

	__forceinline__ __host__ __device__
	volatile f_Average_f& op(volatile f_Average_f& x) volatile
	{
		m_average += x.m_average;
		return *this;
	}

	__forceinline__ __host__ __device__
	f_Average_f& operator=(const f_Average_f& x)
	{
		m_average = x.m_average;
		return *this;
	}

	__forceinline__ __host__ __device__
	volatile f_Average_f& operator=(volatile f_Average_f& x) volatile
	{
		m_average = x.m_average;
		return *this;
	}

	__forceinline__ __host__ __device__
	void finalize(f_Average_f* x) volatile
	{
		x->m_average = m_average;
	}

	static __host__
	void simulate(f_Average_f* out, float* in, int size, int outOff, int inOff, int stride)
	{
		out[outOff].m_average = 0;
		for (int i = inOff; i < inOff + size; i += stride)
		{
			out[outOff].m_average += in[i] / (float)gridDim.x;
		}
	}

	static __host__
	bool check(f_Average_f* x, f_Average_f* y, int outOff, float* in)
	{
		printf("average: %f == %f\n", x[outOff].m_average, y[outOff].m_average);
		return FloatEquals(x[outOff].m_average, y[outOff].m_average);
	}
};

const int f_Average_f::outSize = 4;

template<>
struct SharedMemory<f_Average_f>
{
	__device__ inline operator f_Average_f*()
	{
		extern __shared__ int __f_Average_f[];
		return (f_Average_f*)(void*)__f_Average_f;
	}

	__device__ inline operator f_Average_f*() const
	{
		extern __shared__ int __f_Average_f[];
		return (f_Average_f*)(void*)__f_Average_f;
	}
};

#pragma once

#include <cfloat>
#include <cmath>

#include "SharedMemory.cuh"

class c_Average_c
{
public:
	static const int outSize;

	Complex m_average;

	__forceinline__ __host__ __device__
		c_Average_c& op(Complex x, int idx)
	{
		m_average.R += x.R / (float)gridDim.x;
		m_average.I += x.I / (float)gridDim.x;
		return *this;
	}

	__forceinline__ __host__ __device__
		volatile c_Average_c& op(Complex x, int idx) volatile
	{
		m_average.R += x.R / (float)gridDim.x;
		m_average.I += x.I / (float)gridDim.x;
		return *this;
	}

	__forceinline__ __host__ __device__
		c_Average_c& op(const c_Average_c& x)
	{
		m_average.R += x.m_average.R;
		m_average.I += x.m_average.I;
		return *this;
	}

	__forceinline__ __host__ __device__
		volatile c_Average_c& op(volatile c_Average_c& x) volatile
	{
		m_average.R += x.m_average.R;
		m_average.I += x.m_average.I;
		return *this;
	}

	__forceinline__ __host__ __device__
		c_Average_c& operator=(const c_Average_c& x)
	{
		m_average.R = x.m_average.R;
		m_average.I = x.m_average.I;
		return *this;
	}

	__forceinline__ __host__ __device__
		volatile c_Average_c& operator=(volatile c_Average_c& x) volatile
	{
		m_average.R = x.m_average.R;
		m_average.I = x.m_average.I;
		return *this;
	}

	__forceinline__ __host__ __device__
		void finalize(c_Average_c* x) volatile
	{
		x->m_average.R = m_average.R;
		x->m_average.I = m_average.I;
	}

	static __host__
		void simulate(c_Average_c* out, Complex* in, int size, int outOff, int inOff, int stride)
	{
		out[outOff].m_average.R = 0;
		out[outOff].m_average.I = 0;
		for (int i = inOff; i < inOff + size; i += stride)
		{
			out[outOff].m_average.R += in[i].R / (float)gridDim.x;
			out[outOff].m_average.I += in[i].I / (float)gridDim.x;
		}
	}

	static __host__
		bool check(c_Average_c* x, c_Average_c* y, int outOff, Complex* in)
	{
		printf("average: [%f, %f] == [%f, %f]\n", x[outOff].m_average.R, x[outOff].m_average.I, y[outOff].m_average.R, y[outOff].m_average.I);
		return ComplexEquals(x[outOff].m_average, y[outOff].m_average);
	}
};

const int c_Average_c::outSize = 8;

template<>
struct SharedMemory<c_Average_c>
{
	__device__ inline operator c_Average_c*()
	{
		extern __shared__ int __c_Average_c[];
		return (c_Average_c*)(void*)__c_Average_c;
	}

	__device__ inline operator c_Average_c*() const
	{
		extern __shared__ int __c_Average_c[];
		return (c_Average_c*)(void*)__c_Average_c;
	}
};

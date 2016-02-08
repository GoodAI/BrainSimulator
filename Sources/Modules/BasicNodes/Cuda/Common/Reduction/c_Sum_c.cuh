#pragma once

#include <cfloat>
#include <cmath>

#include "SharedMemory.cuh"

class c_Sum_c
{
public:
	static const int outSize;

	Complex m_sum;

	__forceinline__ __host__ __device__
		c_Sum_c& op(Complex x, int idx)
	{
		m_sum.R += x.R;
		m_sum.I += x.I;
		return *this;
	}

	__forceinline__ __host__ __device__
		volatile c_Sum_c& op(Complex x, int idx) volatile
	{
		m_sum.R += x.R;
		m_sum.I += x.I;
		return *this;
	}

	__forceinline__ __host__ __device__
		c_Sum_c& op(const c_Sum_c& x)
	{
		m_sum.R += x.m_sum.R;
		m_sum.I += x.m_sum.I;
		return *this;
	}

	__forceinline__ __host__ __device__
		volatile c_Sum_c& op(volatile c_Sum_c& x) volatile
	{
		m_sum.R += x.m_sum.R;
		m_sum.I += x.m_sum.I;
		return *this;
	}

	__forceinline__ __host__ __device__
		c_Sum_c& operator=(const c_Sum_c& x)
	{
		m_sum.R = x.m_sum.R;
		m_sum.I = x.m_sum.I;
		return *this;
	}

	__forceinline__ __host__ __device__
		volatile c_Sum_c& operator=(volatile c_Sum_c& x) volatile
	{
		m_sum.R = x.m_sum.R;
		m_sum.I = x.m_sum.I;
		return *this;
	}

	__forceinline__ __host__ __device__
		void finalize(c_Sum_c* x) volatile
	{
		x->m_sum.R = m_sum.R;
		x->m_sum.I = m_sum.I;
	}

	static __host__
		void simulate(c_Sum_c* out, Complex* in, int size, int outOff, int inOff, int stride)
	{
		out[outOff].m_sum.R = 0;
		out[outOff].m_sum.I = 0;
		for (int i = inOff; i < inOff + size; i += stride)
		{
			out[outOff].m_sum.R += in[i].R;
			out[outOff].m_sum.I += in[i].I;
		}
	}

	static __host__
		bool check(c_Sum_c* x, c_Sum_c* y, int outOff, Complex* in)
	{
		printf("sum: [%f, %f] == [%f, %f]\n", x[outOff].m_sum.R, x[outOff].m_sum.I, y[outOff].m_sum.R, y[outOff].m_sum.I);
		return ComplexEquals(x[outOff].m_sum, y[outOff].m_sum);
	}
};

const int c_Sum_c::outSize = 8;

template<>
struct SharedMemory<c_Sum_c>
{
	__device__ inline operator c_Sum_c*()
	{
		extern __shared__ int __c_Sum_c[];
		return (c_Sum_c*)(void*)__c_Sum_c;
	}

	__device__ inline operator c_Sum_c*() const
	{
		extern __shared__ int __c_Sum_c[];
		return (c_Sum_c*)(void*)__c_Sum_c;
	}
};

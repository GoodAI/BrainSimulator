#pragma once

#include <cfloat>
#include <cmath>

#include "SharedMemory.cuh"

class c_ComplexDot_c
{
public:
	static const int outSize;

	Complex m_dot;

	__forceinline__ __host__ __device__
	c_ComplexDot_c& op(Complex x, Complex y, int idx)
	{
		m_dot.R += x.R * y.R - x.I * y.I;
		m_dot.I += x.R * y.I + x.I * y.R;
		return *this;
	}

	__forceinline__ __host__ __device__
	volatile c_ComplexDot_c& op(Complex x, Complex y, int idx) volatile
	{
		m_dot.R += x.R * y.R - x.I * y.I;
		m_dot.I += x.R * y.I + x.I * y.R;
		return *this;
	}

	__forceinline__ __host__ __device__
	c_ComplexDot_c& op(const c_ComplexDot_c& x)
	{
		m_dot.R += x.m_dot.R;
		m_dot.I += x.m_dot.I;
		return *this;
	}

	__forceinline__ __host__ __device__
	volatile c_ComplexDot_c& op(volatile c_ComplexDot_c& x) volatile
	{
		m_dot.R += x.m_dot.R;
		m_dot.I += x.m_dot.I;
		return *this;
	}

	__forceinline__ __host__ __device__
	c_ComplexDot_c& operator=(const c_ComplexDot_c& x)
	{
		m_dot.R = x.m_dot.R;
		m_dot.I = x.m_dot.I;
		return *this;
	}

	__forceinline__ __host__ __device__
	volatile c_ComplexDot_c& operator=(volatile c_ComplexDot_c& x) volatile
	{
		m_dot.R = x.m_dot.R;
		m_dot.I = x.m_dot.I;
		return *this;
	}

	__forceinline__ __host__ __device__
	void finalize(c_ComplexDot_c* x) volatile
	{
		x->m_dot.R = m_dot.R;
		x->m_dot.I = m_dot.I;
	}

	static __host__
	void simulate(c_ComplexDot_c* out, int outOff, Complex* in1, Complex* in2, int size)
	{
		out[outOff].m_dot.R = 0;
		out[outOff].m_dot.I = 0;
		for (int i = 0; i < size; ++i)
		{
			out[outOff].m_dot.R += in1[i].R * in2[i].R - in1[i].I * in2[i].I;
			out[outOff].m_dot.I += in1[i].R * in2[i].I + in1[i].I * in2[i].R;
		}
	}

	static __host__
	bool check(c_ComplexDot_c* x, c_ComplexDot_c* y, int outOff, Complex* in1, Complex* in2)
	{
		printf("dot: [%f, %f] == [%f, %f]\n", x[outOff].m_dot.R, x[outOff].m_dot.I, y[outOff].m_dot.R, y[outOff].m_dot.I);
		return ComplexEquals(x[outOff].m_dot, y[outOff].m_dot);
	}
};

const int c_ComplexDot_c::outSize = 8;

template<>
struct SharedMemory<c_ComplexDot_c>
{
	__device__ inline operator c_ComplexDot_c*()
	{
		extern __shared__ int c_f_ComplexDot_c[];
		return (c_ComplexDot_c*)(void*)c_f_ComplexDot_c;
	}

	__device__ inline operator c_ComplexDot_c*() const
	{
		extern __shared__ int c_f_ComplexDot_c[];
		return (c_ComplexDot_c*)(void*)c_f_ComplexDot_c;
	}
};

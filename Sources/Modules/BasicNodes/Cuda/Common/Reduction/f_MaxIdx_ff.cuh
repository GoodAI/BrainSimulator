#pragma once

#include <cfloat>

#include "SharedMemory.cuh"

class f_MaxIdx_ff
{
public:
	static const int outSize;

	float m_max;
	float m_idx;

	__forceinline__ __host__ __device__
		f_MaxIdx_ff()
	{
		m_max = -FLT_MAX;
		m_idx = -1.0f;
	}

	__forceinline__ __host__ __device__
		f_MaxIdx_ff& op(float x, float idx)
	{
		if (x > m_max)
		{
			m_max = x;
			m_idx = idx;
		}
		return *this;
	}

	__forceinline__ __host__ __device__
		volatile f_MaxIdx_ff& op(float x, float idx) volatile
	{
		if (x > m_max)
		{
			m_max = x;
			m_idx = idx;
		}
		return *this;
	}

	__forceinline__ __host__ __device__
		f_MaxIdx_ff& op(const f_MaxIdx_ff& x)
	{
		if (x.m_max > m_max)
		{
			m_max = x.m_max;
			m_idx = x.m_idx;
		}
		return *this;
	}

	__forceinline__ __host__ __device__
		volatile f_MaxIdx_ff& op(volatile f_MaxIdx_ff& x) volatile
	{
		if (x.m_max > m_max)
		{
			m_max = x.m_max;
			m_idx = x.m_idx;
		}
		return *this;
	}

	__forceinline__ __host__ __device__
		f_MaxIdx_ff& operator=(const f_MaxIdx_ff& x)
	{
		m_max = x.m_max;
		m_idx = x.m_idx;
		return *this;
	}

	__forceinline__ __host__ __device__
		volatile f_MaxIdx_ff& operator=(volatile f_MaxIdx_ff& x) volatile
	{
		m_max = x.m_max;
		m_idx = x.m_idx;
		return *this;
	}

	__forceinline__ __host__ __device__
		void finalize(f_MaxIdx_ff* x) volatile
	{
		x->m_max = m_max;
		x->m_idx = m_idx;
	}

	static __host__
		void simulate(f_MaxIdx_ff* out, float* in, int size, int outOff, int inOff, int stride)
	{
		out[outOff].m_max = in[inOff];
		out[outOff].m_idx = inOff;
		for (int i = inOff; i < inOff + size; i += stride)
		{
			if (in[i] > out[outOff].m_max)
			{
				out[outOff].m_max = in[i];
				out[outOff].m_idx = (float)i;
			}
		}
	}

	static __host__
		bool check(f_MaxIdx_ff* x, f_MaxIdx_ff* y, int outOff, float* in)
	{
		printf("min: %f == %f, idx: in[%d]:%f == in[%d]:%f\n",
			x[outOff].m_max, y[outOff].m_max, x[outOff].m_idx, in[(int)x[outOff].m_idx], y[outOff].m_idx, in[(int)y[outOff].m_idx]);
		return FloatEquals(x[outOff].m_max, y[outOff].m_max) && FloatEquals(in[(int)x[outOff].m_idx], in[(int)y[outOff].m_idx]);
	}
};

const int f_MaxIdx_ff::outSize = 8;

template<>
struct SharedMemory<f_MaxIdx_ff>
{
	__device__ inline operator f_MaxIdx_ff*()
	{
		extern __shared__ int __f_MaxIdx_ff[];
		return (f_MaxIdx_ff*)(void*)__f_MaxIdx_ff;
	}

	__device__ inline operator f_MaxIdx_ff*() const
	{
		extern __shared__ int __f_MaxIdx_ff[];
		return (f_MaxIdx_ff*)(void*)__f_MaxIdx_ff;
	}
};

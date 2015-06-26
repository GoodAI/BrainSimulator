#pragma once

#include <climits>

#include "SharedMemory.cuh"

class i_MaxIdx_2i
{
public:
    static const int outSize;
    
    int m_max;
    int m_idx;

    __forceinline__ __host__ __device__
    i_MaxIdx_2i()
    {
        m_max = INT_MIN;
        m_idx = -1;
    }

    __forceinline__ __host__ __device__
    i_MaxIdx_2i& op(int x, int idx)
    {
        if (x > m_max)
        {
            m_max = x;
            m_idx = idx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile i_MaxIdx_2i& op(int x, int idx) volatile
    {
        if (x > m_max)
        {
            m_max = x;
            m_idx = idx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    i_MaxIdx_2i& op(const i_MaxIdx_2i& x)
    {
        if (x.m_max > m_max)
        {
            m_max = x.m_max;
            m_idx = x.m_idx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile i_MaxIdx_2i& op(volatile i_MaxIdx_2i& x) volatile
    {
        if (x.m_max > m_max)
        {
            m_max = x.m_max;
            m_idx = x.m_idx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    i_MaxIdx_2i& operator=(const i_MaxIdx_2i& x)
    {
        m_max = x.m_max;
        m_idx = x.m_idx;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile i_MaxIdx_2i& operator=(volatile i_MaxIdx_2i& x) volatile
    {
        m_max = x.m_max;
        m_idx = x.m_idx;
        return *this;
    }

    __forceinline__ __host__ __device__
    void finalize(i_MaxIdx_2i* x) volatile
    {
        x->m_max = m_max;
        x->m_idx = m_idx;
    }

    static __host__
    void simulate(i_MaxIdx_2i* out, int* in, int size, int outOff, int inOff, int stride)
    {
        out[outOff].m_max = in[inOff];
        out[outOff].m_idx = inOff;
        for (int i = inOff; i < inOff + size; i += stride)
        {
            if (in[i] > out[outOff].m_max)
            {
                out[outOff].m_max = in[i];
                out[outOff].m_idx = i;
            }
        }
    }

    static __host__
    bool check(i_MaxIdx_2i* x, i_MaxIdx_2i* y, int outOff, int* in)
    {
        printf("max: %d == %d, idx: in[%d]:%d == in[%d]:%d\n",
            x[outOff].m_max, y[outOff].m_max, x[outOff].m_idx, in[x[outOff].m_idx], y[outOff].m_idx, in[y[outOff].m_idx]);
        return x[outOff].m_max == y[outOff].m_max && in[x[outOff].m_idx] == in[y[outOff].m_idx];
    }
};

const int i_MaxIdx_2i::outSize = 8;

template<>
struct SharedMemory<i_MaxIdx_2i>
{
    __device__ inline operator i_MaxIdx_2i*()
    {
        extern __shared__ int __i_MaxIdx_2i[];
        return (i_MaxIdx_2i*)(void*)__i_MaxIdx_2i;
    }

    __device__ inline operator i_MaxIdx_2i*() const
    {
        extern __shared__ int __i_MaxIdx_2i[];
        return (i_MaxIdx_2i*)(void*)__i_MaxIdx_2i;
    }
};

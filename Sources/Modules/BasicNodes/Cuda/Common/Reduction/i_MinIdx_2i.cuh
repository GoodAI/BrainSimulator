#pragma once

#include <climits>

#include "SharedMemory.cuh"

class i_MinIdx_2i
{
public:
    static const int outSize;
    
    int m_min;
    int m_idx;

    __forceinline__ __host__ __device__
    i_MinIdx_2i()
    {
        m_min = INT_MAX;
        m_idx = -1;
    }

    __forceinline__ __host__ __device__
    i_MinIdx_2i& op(int x, int idx)
    {
        if (x < m_min)
        {
            m_min = x;
            m_idx = idx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile i_MinIdx_2i& op(int x, int idx) volatile
    {
        if (x < m_min)
        {
            m_min = x;
            m_idx = idx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    i_MinIdx_2i& op(const i_MinIdx_2i& x)
    {
        if (x.m_min < m_min)
        {
            m_min = x.m_min;
            m_idx = x.m_idx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile i_MinIdx_2i& op(volatile i_MinIdx_2i& x) volatile
    {
        if (x.m_min < m_min)
        {
            m_min = x.m_min;
            m_idx = x.m_idx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    i_MinIdx_2i& operator=(const i_MinIdx_2i& x)
    {
        m_min = x.m_min;
        m_idx = x.m_idx;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile i_MinIdx_2i& operator=(volatile i_MinIdx_2i& x) volatile
    {
        m_min = x.m_min;
        m_idx = x.m_idx;
        return *this;
    }

    __forceinline__ __host__ __device__
    void finalize(i_MinIdx_2i* x) volatile
    {
        x->m_min = m_min;
        x->m_idx = m_idx;
    }

    static __host__
    void simulate(i_MinIdx_2i* out, int* in, int size, int outOff, int inOff, int stride)
    {
        out[outOff].m_min = in[inOff];
        out[outOff].m_idx = inOff;
        for (int i = inOff; i < inOff + size; i += stride)
        {
            if (in[i] < out[outOff].m_min)
            {
                out[outOff].m_min = in[i];
                out[outOff].m_idx = i;
            }
        }
    }

    static __host__
    bool check(i_MinIdx_2i* x, i_MinIdx_2i* y, int outOff, int* in)
    {
        printf("min: %d == %d, idx: in[%d]:%d == in[%d]:%d\n",
            x[outOff].m_min, y[outOff].m_min, x[outOff].m_idx, in[x[outOff].m_idx], y[outOff].m_idx, in[y[outOff].m_idx]);
        return x[outOff].m_min == y[outOff].m_min && in[x[outOff].m_idx] == in[y[outOff].m_idx];
    }
};

const int i_MinIdx_2i::outSize = 8;

template<>
struct SharedMemory<i_MinIdx_2i>
{
    __forceinline__ __device__ inline operator i_MinIdx_2i*()
    {
        extern __shared__ int __i_MinIdx_2i[];
        return (i_MinIdx_2i*)(void*)__i_MinIdx_2i;
    }

    __forceinline__ __device__ inline operator i_MinIdx_2i*() const
    {
        extern __shared__ int __i_MinIdx_2i[];
        return (i_MinIdx_2i*)(void*)__i_MinIdx_2i;
    }
};

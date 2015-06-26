#pragma once

#include <limits.h>

#include "SharedMemory.cuh"

class i_MinIdxMaxIdx_4i
{
public:
    static const int outSize;
    
    int m_min;
    int m_minIdx;
    int m_max;
    int m_maxIdx;

    __forceinline__ __host__ __device__
    i_MinIdxMaxIdx_4i()
    {
        m_min = INT_MAX;
        m_minIdx = -1;
        m_max = INT_MIN;
        m_maxIdx = -1;
    }

    __forceinline__ __host__ __device__
    i_MinIdxMaxIdx_4i& op(int x, int idx)
    {
        if (x < m_min)
        {
            m_min = x;
            m_minIdx = idx;
        }
        if (x > m_max)
        {
            m_max = x;
            m_maxIdx = idx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile i_MinIdxMaxIdx_4i& op(int x, int idx) volatile
    {
        if (x < m_min)
        {
            m_min = x;
            m_minIdx = idx;
        }
        if (x > m_max)
        {
            m_max = x;
            m_maxIdx = idx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    i_MinIdxMaxIdx_4i& op(const i_MinIdxMaxIdx_4i& x)
    {
        if (x.m_min < m_min)
        {
            m_min = x.m_min;
            m_minIdx = x.m_minIdx;
        }
        if (x.m_max > m_max)
        {
            m_max = x.m_max;
            m_maxIdx = x.m_maxIdx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile i_MinIdxMaxIdx_4i& op(volatile i_MinIdxMaxIdx_4i& x) volatile
    {
        if (x.m_min < m_min)
        {
            m_min = x.m_min;
            m_minIdx = x.m_minIdx;
        }
        if (x.m_max > m_max)
        {
            m_max = x.m_max;
            m_maxIdx = x.m_maxIdx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    i_MinIdxMaxIdx_4i& operator=(const i_MinIdxMaxIdx_4i& x)
    {
        m_min = x.m_min;
        m_minIdx = x.m_minIdx;
        m_max = x.m_max;
        m_maxIdx = x.m_maxIdx;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile i_MinIdxMaxIdx_4i& operator=(volatile i_MinIdxMaxIdx_4i& x) volatile
    {
        m_min = x.m_min;
        m_minIdx = x.m_minIdx;
        m_max = x.m_max;
        m_maxIdx = x.m_maxIdx;
        return *this;
    }

    __forceinline__ __host__ __device__
    void finalize(i_MinIdxMaxIdx_4i* x) volatile
    {
        x->m_min = m_min;
        x->m_minIdx = m_minIdx;
        x->m_max = m_max;
        x->m_maxIdx = m_maxIdx;
    }

    static __host__
    void simulate(i_MinIdxMaxIdx_4i* out, int* in, int size, int outOff, int inOff, int stride)
    {
        out[outOff].m_min = in[inOff];
        out[outOff].m_minIdx = inOff;
        out[outOff].m_max = in[inOff];
        out[outOff].m_maxIdx = inOff;
        for (int i = inOff; i < inOff + size; i += stride)
        {
            if (in[i] < out[outOff].m_min)
            {
                out[outOff].m_min = in[i];
                out[outOff].m_minIdx = i;
            }
            if (in[i] > out[outOff].m_max)
            {
                out[outOff].m_max = in[i];
                out[outOff].m_maxIdx = i;
            }
        }
    }

    static __host__
    bool check(i_MinIdxMaxIdx_4i* x, i_MinIdxMaxIdx_4i* y, int outOff, int* in)
    {
        printf("max: %d == %d, maxIdx: in[%d]:%d == in[%d]:%d, min: %d == %d, minIdx: in[%d]:%d == in[%d]:%d\n",
            x[outOff].m_max, y[outOff].m_max, x[outOff].m_maxIdx, in[x[outOff].m_maxIdx], y[outOff].m_maxIdx, in[y[outOff].m_maxIdx], x[outOff].m_min, y[outOff].m_min, x[outOff].m_minIdx, in[x[outOff].m_minIdx], y[outOff].m_minIdx, in[y[outOff].m_minIdx]);
        return x[outOff].m_max == y[outOff].m_max && in[x[outOff].m_maxIdx] == in[y[outOff].m_maxIdx]
            && x[outOff].m_min == y[outOff].m_min && in[x[outOff].m_minIdx] == in[y[outOff].m_minIdx];
    }
};

const int i_MinIdxMaxIdx_4i::outSize = 16;

template<>
struct SharedMemory<i_MinIdxMaxIdx_4i>
{
    __device__ inline operator i_MinIdxMaxIdx_4i*()
    {
        extern __shared__ int __i_MinIdxMaxIdx_4i[];
        return (i_MinIdxMaxIdx_4i*)(void*)__i_MinIdxMaxIdx_4i;
    }

    __device__ inline operator i_MinIdxMaxIdx_4i*() const
    {
        extern __shared__ int __i_MinIdxMaxIdx_4i[];
        return (i_MinIdxMaxIdx_4i*)(void*)__i_MinIdxMaxIdx_4i;
    }
};

#pragma once

#include <float.h>

#include "SharedMemory.cuh"

class f_MinIdxMaxIdx_fifi
{
public:
    static const int outSize;
    
    float m_min;
    int m_minIdx;
    float m_max;
    int m_maxIdx;

    __forceinline__ __host__ __device__
    f_MinIdxMaxIdx_fifi()
    {
        m_min = FLT_MAX;
        m_minIdx = -1;
        m_max = -FLT_MAX;
        m_maxIdx = -1;
    }

    __forceinline__ __host__ __device__
    f_MinIdxMaxIdx_fifi& op(float x, int idx)
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
    volatile f_MinIdxMaxIdx_fifi& op(float x, int idx) volatile
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
    f_MinIdxMaxIdx_fifi& op(const f_MinIdxMaxIdx_fifi& x)
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
    volatile f_MinIdxMaxIdx_fifi& op(volatile f_MinIdxMaxIdx_fifi& x) volatile
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
    f_MinIdxMaxIdx_fifi& operator=(const f_MinIdxMaxIdx_fifi& x)
    {
        m_min = x.m_min;
        m_minIdx = x.m_minIdx;
        m_max = x.m_max;
        m_maxIdx = x.m_maxIdx;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_MinIdxMaxIdx_fifi& operator=(volatile f_MinIdxMaxIdx_fifi& x) volatile
    {
        m_min = x.m_min;
        m_minIdx = x.m_minIdx;
        m_max = x.m_max;
        m_maxIdx = x.m_maxIdx;
        return *this;
    }

    __forceinline__ __host__ __device__
    void finalize(f_MinIdxMaxIdx_fifi* x) volatile
    {
        x->m_min = m_min;
        x->m_minIdx = m_minIdx;
        x->m_max = m_max;
        x->m_maxIdx = m_maxIdx;
    }

    static __host__
    void simulate(f_MinIdxMaxIdx_fifi* out, float* in, int size, int outOff, int inOff, int stride)
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
    bool check(f_MinIdxMaxIdx_fifi* x, f_MinIdxMaxIdx_fifi* y, int outOff, float* in)
    {
        printf("max: %f == %f, maxIdx: in[%d]:%f == in[%d]:%f, min: %f == %f, minIdx: in[%d]:%f == in[%d]:%f\n",
            x[outOff].m_max, y[outOff].m_max, x[outOff].m_maxIdx, in[x[outOff].m_maxIdx], y[outOff].m_maxIdx, in[y[outOff].m_maxIdx], x[outOff].m_min, y[outOff].m_min, x[outOff].m_minIdx, in[x[outOff].m_minIdx], y[outOff].m_minIdx, in[y[outOff].m_minIdx]);
        return FloatEquals(x[outOff].m_max, y[outOff].m_max) && FloatEquals(in[x[outOff].m_maxIdx], in[y[outOff].m_maxIdx])
            && FloatEquals(x[outOff].m_min, y[outOff].m_min) && FloatEquals(in[x[outOff].m_minIdx], in[y[outOff].m_minIdx]);
    }
};

const int f_MinIdxMaxIdx_fifi::outSize = 16;

template<>
struct SharedMemory<f_MinIdxMaxIdx_fifi>
{
    __device__ inline operator f_MinIdxMaxIdx_fifi*()
    {
        extern __shared__ int __f_MinIdxMaxIdx_fifi[];
        return (f_MinIdxMaxIdx_fifi*)(void*)__f_MinIdxMaxIdx_fifi;
    }

    __device__ inline operator f_MinIdxMaxIdx_fifi*() const
    {
        extern __shared__ int __f_MinIdxMaxIdx_fifi[];
        return (f_MinIdxMaxIdx_fifi*)(void*)__f_MinIdxMaxIdx_fifi;
    }
};

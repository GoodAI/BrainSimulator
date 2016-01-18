#pragma once

#include <float.h>

#include "SharedMemory.cuh"

class f_MinIdx_fi
{
public:
    static const int outSize;
    
    float m_min;
    int m_idx;

    __forceinline__ __host__ __device__
    f_MinIdx_fi()
    {
        m_min = FLT_MAX;
        m_idx = -1;
    }

    __forceinline__ __host__ __device__
    f_MinIdx_fi& op(float x, int idx)
    {
        if (x < m_min)
        {
            m_min = x;
            m_idx = idx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_MinIdx_fi& op(float x, int idx) volatile
    {
        if (x < m_min)
        {
            m_min = x;
            m_idx = idx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    f_MinIdx_fi& op(const f_MinIdx_fi& x)
    {
        if (x.m_min < m_min)
        {
            m_min = x.m_min;
            m_idx = x.m_idx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_MinIdx_fi& op(volatile f_MinIdx_fi& x) volatile
    {
        if (x.m_min < m_min)
        {
            m_min = x.m_min;
            m_idx = x.m_idx;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    f_MinIdx_fi& operator=(const f_MinIdx_fi& x)
    {
        m_min = x.m_min;
        m_idx = x.m_idx;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_MinIdx_fi& operator=(volatile f_MinIdx_fi& x) volatile
    {
        m_min = x.m_min;
        m_idx = x.m_idx;
        return *this;
    }

    __forceinline__ __host__ __device__
    void finalize(f_MinIdx_fi* x) volatile
    {
        x->m_min = m_min;
        x->m_idx = m_idx;
    }

    static __host__
    void simulate(f_MinIdx_fi* out, float* in, int size, int outOff, int inOff, int stride)
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
    bool check(f_MinIdx_fi* x, f_MinIdx_fi* y, int outOff, float* in)
    {
        printf("min: %f == %f, idx: in[%d]:%f == in[%d]:%f\n",
            x[outOff].m_min, y[outOff].m_min, x[outOff].m_idx, in[x[outOff].m_idx], y[outOff].m_idx, in[y[outOff].m_idx]);
        return FloatEquals(x[outOff].m_min, y[outOff].m_min) && FloatEquals(in[x[outOff].m_idx], in[y[outOff].m_idx]);
    }
};

const int f_MinIdx_fi::outSize = 8;

template<>
struct SharedMemory<f_MinIdx_fi>
{
    __device__ inline operator f_MinIdx_fi*()
    {
        extern __shared__ int __f_MinIdx_fi[];
        return (f_MinIdx_fi*)(void*)__f_MinIdx_fi;
    }

    __device__ inline operator f_MinIdx_fi*() const
    {
        extern __shared__ int __f_MinIdx_fi[];
        return (f_MinIdx_fi*)(void*)__f_MinIdx_fi;
    }
};

#pragma once

#include <cfloat>
#include <cmath>

#include "SharedMemory.cuh"

class f_MinMax_2f
{
public:
    static const int outSize;
    
    float m_min;
    float m_max;

    __forceinline__ __host__ __device__
    f_MinMax_2f()
    {
        m_min = FLT_MAX;
        m_max = -FLT_MAX;
    }

    __forceinline__ __host__ __device__
    f_MinMax_2f& op(float x, int idx)
    {
        if (x < m_min)
        {
            m_min = x;
        }
        if (x > m_max)
        {
            m_max = x;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_MinMax_2f& op(float x, int idx) volatile
    {
        if (x < m_min)
        {
            m_min = x;
        }
        if (x > m_max)
        {
            m_max = x;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    f_MinMax_2f& op(const f_MinMax_2f& x)
    {
        if (x.m_min < m_min)
        {
            m_min = x.m_min;
        }
        if (x.m_max > m_max)
        {
            m_max = x.m_max;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_MinMax_2f& op(volatile f_MinMax_2f& x) volatile
    {
        if (x.m_min < m_min)
        {
            m_min = x.m_min;
        }
        if (x.m_max > m_max)
        {
            m_max = x.m_max;
        }
        return *this;
    }

    __forceinline__ __host__ __device__
    f_MinMax_2f& operator=(const f_MinMax_2f& x)
    {
        m_min = x.m_min;
        m_max = x.m_max;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_MinMax_2f& operator=(volatile f_MinMax_2f& x) volatile
    {
        m_min = x.m_min;
        m_max = x.m_max;
        return *this;
    }

    __forceinline__ __host__ __device__
    void finalize(f_MinMax_2f* x) volatile
    {
        x->m_min = m_min;
        x->m_max = m_max;
    }

    static __host__
    void simulate(f_MinMax_2f* out, float* in, int size, int outOff, int inOff, int stride)
    {
        out[outOff].m_min = in[inOff];
        out[outOff].m_max = in[inOff];
        for (int i = inOff; i < inOff + size; i += stride)
        {
            if (in[i] < out[outOff].m_min)
            {
                out[outOff].m_min = in[i];
            }
            if (in[i] > out[outOff].m_max)
            {
                out[outOff].m_max = in[i];
            }
        }
    }

    static __host__
    bool check(f_MinMax_2f* x, f_MinMax_2f* y, int outOff, float* in)
    {
        printf("max: %f == %f, min: %f == %f\n", x[outOff].m_max, y[outOff].m_max, x[outOff].m_min, y[outOff].m_min);
        return FloatEquals(x[outOff].m_max, y[outOff].m_max) && FloatEquals(x[outOff].m_min, y[outOff].m_min);
    }
};

const int f_MinMax_2f::outSize = 8;

template<>
struct SharedMemory<f_MinMax_2f>
{
    __device__ inline operator f_MinMax_2f*()
    {
        extern __shared__ int __f_MinMax_2f[];
        return (f_MinMax_2f*)(void*)__f_MinMax_2f;
    }

    __device__ inline operator f_MinMax_2f*() const
    {
        extern __shared__ int __f_MinMax_2f[];
        return (f_MinMax_2f*)(void*)__f_MinMax_2f;
    }
};

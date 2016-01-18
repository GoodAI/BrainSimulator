#pragma once

#include <cfloat>
#include <cmath>

#include "SharedMemory.cuh"

class f_Sum_f
{
public:
    static const int outSize;
    
    float m_sum;

    __forceinline__ __host__ __device__
    f_Sum_f()
    {
        m_sum = 0;
    }

    __forceinline__ __host__ __device__
    f_Sum_f& op(float x, int idx)
    {
        m_sum += x;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_Sum_f& op(float x, int idx) volatile
    {
        m_sum += x;
        return *this;
    }

    __forceinline__ __host__ __device__
    f_Sum_f& op(const f_Sum_f& x)
    {
        m_sum += x.m_sum;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_Sum_f& op(volatile f_Sum_f& x) volatile
    {
        m_sum += x.m_sum;
        return *this;
    }

    __forceinline__ __host__ __device__
    f_Sum_f& operator=(const f_Sum_f& x)
    {
        m_sum = x.m_sum;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_Sum_f& operator=(volatile f_Sum_f& x) volatile
    {
        m_sum = x.m_sum;
        return *this;
    }

    __forceinline__ __host__ __device__
    void finalize(f_Sum_f* x) volatile
    {
        x->m_sum = m_sum;
    }

    static __host__
    void simulate(f_Sum_f* out, float* in, int size, int outOff, int inOff, int stride)
    {
        out[outOff].m_sum = 0;
        for (int i = inOff; i < inOff + size; i += stride)
        {
            out[outOff].m_sum += in[i];
        }
    }

    static __host__
    bool check(f_Sum_f* x, f_Sum_f* y, int outOff, float* in)
    {
        printf("sum: %f == %f\n", x[outOff].m_sum, y[outOff].m_sum);
        return FloatEquals(x[outOff].m_sum, y[outOff].m_sum);
    }
};

const int f_Sum_f::outSize = 4;

template<>
struct SharedMemory<f_Sum_f>
{
    __device__ inline operator f_Sum_f*()
    {
        extern __shared__ int __f_Sum_f[];
        return (f_Sum_f*)(void*)__f_Sum_f;
    }

    __device__ inline operator f_Sum_f*() const
    {
        extern __shared__ int __f_Sum_f[];
        return (f_Sum_f*)(void*)__f_Sum_f;
    }
};

#pragma once

#include <cfloat>
#include <cmath>

#include "SharedMemory.cuh"

class f_Dot_f
{
public:
    static const int outSize;
    
    float m_dot;

    __forceinline__ __host__ __device__
    f_Dot_f()
    {
        m_dot = 0;
    }

    __forceinline__ __host__ __device__
    f_Dot_f& op(float x, float y, int idx)
    {
        m_dot += x*y;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_Dot_f& op(float x, float y, int idx) volatile
    {
        m_dot += x*y;
        return *this;
    }

    __forceinline__ __host__ __device__
    f_Dot_f& op(const f_Dot_f& x)
    {
        m_dot += x.m_dot;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_Dot_f& op(volatile f_Dot_f& x) volatile
    {
        m_dot += x.m_dot;
        return *this;
    }

    __forceinline__ __host__ __device__
    f_Dot_f& operator=(const f_Dot_f& x)
    {
        m_dot = x.m_dot;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_Dot_f& operator=(volatile f_Dot_f& x) volatile
    {
        m_dot = x.m_dot;
        return *this;
    }

    __forceinline__ __host__ __device__
    void finalize(f_Dot_f* x) volatile
    {
        x->m_dot = m_dot;
    }

    static __host__
    void simulate(f_Dot_f* out, int outOff, float* in1, float* in2, int size)
    {
        out[outOff].m_dot = 0;
        for (int i = 0; i < size; ++i)
        {
            out[outOff].m_dot += in1[i] * in2[i];
        }
    }

    static __host__
    bool check(f_Dot_f* x, f_Dot_f* y, int outOff, float* in1, float* in2)
    {
        //printf("dot: %f == %f\n", x[outOff].m_dot, y[outOff].m_dot);
        return FloatEquals(x[outOff].m_dot, y[outOff].m_dot);
    }
};

const int f_Dot_f::outSize = 4;

template<>
struct SharedMemory<f_Dot_f>
{
    __device__ inline operator f_Dot_f*()
    {
        extern __shared__ int __f_Dot_f[];
        return (f_Dot_f*)(void*)__f_Dot_f;
    }

    __device__ inline operator f_Dot_f*() const
    {
        extern __shared__ int __f_Dot_f[];
        return (f_Dot_f*)(void*)__f_Dot_f;
    }
};

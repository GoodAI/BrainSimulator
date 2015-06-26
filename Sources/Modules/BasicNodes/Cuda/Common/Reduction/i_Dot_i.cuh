#pragma once

#include <cfloat>
#include <cmath>

#include "SharedMemory.cuh"

class i_Dot_i
{
public:
    static const int outSize;

    int m_dot;

    __forceinline__ __host__ __device__
    i_Dot_i()
    {
        m_dot = 0;
    }

    __forceinline__ __host__ __device__
    i_Dot_i& op(int x, int y, int idx)
    {
        m_dot += x*y;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile i_Dot_i& op(int x, int y, int idx) volatile
    {
        m_dot += x*y;
        return *this;
    }

    __forceinline__ __host__ __device__
    i_Dot_i& op(const i_Dot_i& x)
    {
        m_dot += x.m_dot;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile i_Dot_i& op(volatile i_Dot_i& x) volatile
    {
        m_dot += x.m_dot;
        return *this;
    }

    __forceinline__ __host__ __device__
    i_Dot_i& operator=(const i_Dot_i& x)
    {
        m_dot = x.m_dot;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile i_Dot_i& operator=(volatile i_Dot_i& x) volatile
    {
        m_dot = x.m_dot;
        return *this;
    }

    __forceinline__ __host__ __device__
    void finalize(i_Dot_i* x) volatile
    {
        x->m_dot = m_dot;
    }

    static __host__
    void simulate(i_Dot_i* out, int outOff, int* in1, int* in2, int size)
    {
        out[outOff].m_dot = 0;
        for (int i = 0; i < size; ++i)
        {
            out[outOff].m_dot += in1[i] * in2[i];
        }
    }

    static __host__
    bool check(i_Dot_i* x, i_Dot_i* y, int outOff, int* in1, int* in2)
    {
        printf("dot: %d == %d\n", x[outOff].m_dot, y[outOff].m_dot);
        return x[outOff].m_dot == y[outOff].m_dot;
    }
};

const int i_Dot_i::outSize = 4;

template<>
struct SharedMemory<i_Dot_i>
{
    __device__ inline operator i_Dot_i*()
    {
        extern __shared__ int __i_Dot_i[];
        return (i_Dot_i*)(void*)__i_Dot_i;
    }

    __device__ inline operator i_Dot_i*() const
    {
        extern __shared__ int __i_Dot_i[];
        return (i_Dot_i*)(void*)__i_Dot_i;
    }
};

#pragma once

#include "SharedMemory.cuh"

class i_Sum_i
{
public:
    static const int outSize;

    int m_sum;

    __forceinline__ __host__ __device__
    i_Sum_i()
    {
        m_sum = 0;
    }

    __forceinline__ __host__ __device__
    i_Sum_i& op(int x, int idx)
    {
        m_sum += x;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile i_Sum_i& op(int x, int idx) volatile
    {
        m_sum += x;
        return *this;
    }

    __forceinline__ __host__ __device__
    i_Sum_i& op(const i_Sum_i& x)
    {
        m_sum += x.m_sum;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile i_Sum_i& op(volatile i_Sum_i& x) volatile
    {
        m_sum += x.m_sum;
        return *this;
    }

    __forceinline__ __host__ __device__
    i_Sum_i& operator=(const i_Sum_i& x)
    {
        m_sum = x.m_sum;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile i_Sum_i& operator=(volatile i_Sum_i& x) volatile
    {
        m_sum = x.m_sum;
        return *this;
    }

    __forceinline__ __host__ __device__
    void finalize(i_Sum_i* x) volatile
    {
        x->m_sum = m_sum;
    }

    static __host__
    void simulate(i_Sum_i* out, int* in, int size, int outOff, int inOff, int stride)
    {
        out[outOff].m_sum = 0;
        for (int i = inOff; i < inOff + size; i += stride)
        {
            out[outOff].m_sum += in[i];
        }
    }

    static __host__
    bool check(i_Sum_i* x, i_Sum_i* y, int outOff, int* in)
    {
        printf("sum: %d == %d\n", x[outOff].m_sum, y[outOff].m_sum);
        return x[outOff].m_sum == y[outOff].m_sum;
    }
};

const int i_Sum_i::outSize = 4;

template<>
struct SharedMemory<i_Sum_i>
{
    __device__ inline operator i_Sum_i*()
    {
        extern __shared__ int __i_Sum_i[];
        return (i_Sum_i*)(void*)__i_Sum_i;
    }

    __device__ inline operator i_Sum_i*() const
    {
        extern __shared__ int __i_Sum_i[];
        return (i_Sum_i*)(void*)__i_Sum_i;
    }
};

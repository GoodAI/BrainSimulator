#pragma once

#include <cmath>

#include "SharedMemory.cuh"

class f_Cosine_f
{
public:
    static const int outSize;
    
    float m_xy;
    float m_x;
    float m_y;
    float m_cos;

    __forceinline__ __host__ __device__
    f_Cosine_f()
    {
        m_xy = 0;
        m_x = 0;
        m_y = 0;
        m_cos = 0;
    }

    __forceinline__ __host__ __device__
    f_Cosine_f& op(float x, float y, int idx)
    {
        m_xy += x*y;
        m_x += x*x;
        m_y += y*y;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_Cosine_f& op(float x, float y, int idx) volatile
    {
        m_xy += x*y;
        m_x += x*x;
        m_y += y*y;
        return *this;
    }

    __forceinline__ __host__ __device__
    f_Cosine_f& op(const f_Cosine_f& x)
    {
        m_xy += x.m_xy;
        m_x += x.m_x;
        m_y += x.m_y;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_Cosine_f& op(volatile f_Cosine_f& x) volatile
    {
        m_xy += x.m_xy;
        m_x += x.m_x;
        m_y += x.m_y;
        return *this;
    }

    __forceinline__ __host__ __device__
    f_Cosine_f& operator=(const f_Cosine_f& x)
    {
        m_xy = x.m_xy;
        m_x = x.m_x;
        m_y = x.m_y;
        return *this;
    }

    __forceinline__ __host__ __device__
    volatile f_Cosine_f& operator=(volatile f_Cosine_f& x) volatile
    {
        m_xy = x.m_xy;
        m_x = x.m_x;
        m_y = x.m_y;
        return *this;
    }

    __forceinline__ __host__ __device__
    void finalize(f_Cosine_f* x) volatile
    {
        float* out = reinterpret_cast<float*>(x);
        out[0] = m_xy / (sqrt(m_x) * sqrt(m_y));
    }

    static __host__
    void simulate(f_Cosine_f* out, int outOff, float* in1, float* in2, int size)
    {
        float c_xy = 0;
        float c_x = 0;
        float c_y = 0;
        for (int i = 0; i < size; ++i)
        {
            c_xy += in1[i] * in2[i];
            c_x += in1[i] * in1[i];
            c_y += in2[i] * in2[i];
        }
        float* x = reinterpret_cast<float*>(reinterpret_cast<char*>(out)+outSize*outOff);
        x[0] = c_xy / (sqrt(c_x) * sqrt(c_y));
    }

    static __host__
    bool check(f_Cosine_f* x, f_Cosine_f* y, int outOff, float* in1, float* in2)
    {
        float* a = reinterpret_cast<float*>(reinterpret_cast<char*>(x)+outSize*outOff);
        float* b = reinterpret_cast<float*>(reinterpret_cast<char*>(y)+outSize*outOff);
        printf("cosine: %f == %f\n", a[0], b[0]);
        return FloatEquals(a[0], b[0]);
    }
};

const int f_Cosine_f::outSize = 4;

template<>
struct SharedMemory<f_Cosine_f>
{
    __device__ inline operator f_Cosine_f*()
    {
        extern __shared__ int __f_Cosine_f[];
        return (f_Cosine_f*)(void*)__f_Cosine_f;
    }

    __device__ inline operator f_Cosine_f*() const
    {
        extern __shared__ int __f_Cosine_f[];
        return (f_Cosine_f*)(void*)__f_Cosine_f;
    }
};

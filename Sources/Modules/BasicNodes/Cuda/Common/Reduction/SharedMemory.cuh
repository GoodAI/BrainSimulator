#pragma once

bool floatEquals(float x, float y)
{
    return fabs(x - y) < (x+y)/(2*100) || fabs(x - y) < 0.01;
}

template<class T>
struct SharedMemory
{
    __device__ inline operator T*()
    {
        extern __shared__ int __smem[];
        return (T*)(void*)__smem;
    }

    __device__ inline operator T*() const
    {
        extern __shared__ int __smem[];
        return (T*)(void*)__smem;
    }
};

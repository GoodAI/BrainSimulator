#pragma once

bool floatEquals(float x, float y)
{
    return fabs(x - y) < (x+y)/(2*100) || fabs(x - y) < 0.01;
}

struct Complex
{
	float R, C;

	__forceinline__ __host__ __device__
	Complex() { R = 0; C = 0; }

	__forceinline__ __host__ __device__
	Complex(float R, float C) : R{R}, C{C} {}

	__forceinline__ __host__ __device__
	Complex(volatile const Complex& rhs) : R{rhs.R}, C{rhs.C} { }
};

bool complexEquals(Complex x, Complex y)
{
	return fabs(x.R - y.R) < (x.R + y.R) / (2 * 100) || fabs(x.R - y.R) < 0.01
		&& fabs(x.C - y.C) < (x.C + y.C) / (2 * 100) || fabs(x.C - y.C) < 0.01;
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

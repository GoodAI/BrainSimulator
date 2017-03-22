#define _SIZE_T_DEFINED 

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>
#include <math.h>

extern "C"
{

    //  DEVICE KERNELS
    __forceinline__ __device__ int GetId()
    {
        return blockDim.x * blockIdx.y * gridDim.x //rows preceeding current row in grid
            + blockDim.x * blockIdx.x             //blocks preceeding current block
            + threadIdx.x;
    }

    __global__ void ChewDataKernel(float a, float* input, float* output, int size, int cycleCountThausands)
    {
        int id = GetId();

        if (id >= size - 1)
            return;

        float x = input[id];
        float y = input[id + 1];
        float z = 0.0f;

        if (cycleCountThausands < 1)
            cycleCountThausands = 1;

        for (int i = 0; i < cycleCountThausands * 1000; i++)
        {
            z = x + a * y + 0.01f * z;
            z = x + sqrtf(z);
            z = z + 0.1f * a * y;
        }

        output[id] = z + a * y;
    }
}

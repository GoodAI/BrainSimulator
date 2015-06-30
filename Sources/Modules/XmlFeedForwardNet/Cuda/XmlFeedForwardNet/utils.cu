#define _SIZE_T_DEFINED 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __cplusplus 
#define __cplusplus 
#endif

#include <cuda.h> 
#include <math_constants.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>


extern "C"  
{
	typedef unsigned int uint;

	struct MyLayerDim
	{
		float* Ptr;     // GPU pointer to the offseted data
        size_t Nb;      // Number of images
        size_t Width;   // Width of each image
        size_t Height;  // Height of each image
        size_t Depth;  // Depth (Used for weights)
        size_t Size;    // Size of each image (Width * Height)
        size_t Count;   // Total number of channels (Nb * Size)
	};

	
	struct KernelDim
    {
        size_t Nb;
        size_t Width;
        size_t Height;
        size_t Depth;
        size_t Size;
        size_t Count;
    };

}
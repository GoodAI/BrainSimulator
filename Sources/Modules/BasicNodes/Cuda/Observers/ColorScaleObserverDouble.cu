#define _SIZE_T_DEFINED 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __cplusplus 
#define __cplusplus 
#endif

#include <cuda.h> 
#include <math_constants.h> 
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_types.h> 
#include <vector_functions.h> 
#include <float.h>

#include "ColorHelpers.cu"

extern "C"  
{	
	//kernel code
	__global__ void ColorScaleObserverDouble(double* values, int method, int scale, float minValue, float maxValue, unsigned int* pixels, int numOfPixels)
	{		
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		if(id < numOfPixels) //id of the thread is valid
		{	
			pixels[id] = float_to_uint_rgba(values[id], method, scale, minValue, maxValue);
		}
	}

   /*
    *   Schematic depicitation of variable's meaning:
    * 
    *           Values:                            Pixels:
    *                                                  <--  tiles in row = 2  -->                                         
    *         <-- tw=3 -->      +                         0  1  2  | 9  10 11
    *            0  1  2        |                         3  4  5  | 12 13 14      tile_row = 0
    *  idtile    3  4  5        | th=3 (tile height)      6  7  8  | 15 16 17
    *    0       6  7  8        |                        -   -   -   -   -   -
    *           -   -   -       +                         18 19 20 | 27 28...
    *            9  10 11                                 21 22 23 |               tile_row = 1
    *  idtile    12 13 14                                 24 25 26 |   ... 35
    *     1      15 16 17 
    *           -   -   -   
    *            18 19 20
    *            21 22 23
    *            24 25 26
    *            .   .
    *            .   .
    *            .
    *         
    */
	__global__ void ColorScaleObserverTiledDouble(double* values, int method, int scale, float minValue, float maxValue, unsigned int* pixels, int numOfPixels, int tw, int th, int tilesInRow)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

        int ta, values_row, values_col, id_tile, tile_row, tile_col, pixels_row, pixels_col;
		if (id < numOfPixels)
		{
            ta           = tw*th;                 // tile area
            values_row   = (id % ta) / tw;        // tile specific row id, in the value smemory block
            values_col   = id % tw;               // tile-specific colum id, in the value smemory block
            id_tile      = id / ta;               // which tile it is
            tile_row     = id_tile % tilesInRow;
            tile_col     = id_tile / tilesInRow;
            pixels_row   = 1;
            pixels_col   = 1;
            
			//pixels[id] = float_to_uint_rgba(values[id], method, scale, minValue, maxValue);
		}
	}

	__global__ void DrawVectorsKernel(float* values, int elements, float maxValue, unsigned int* pixels, int numOfPixels) 
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	
				+ blockDim.x*blockIdx.x				
				+ threadIdx.x;

		if(id < numOfPixels) //id of the thread is valid
		{
			float x = values[id] / maxValue;
			float y = values[numOfPixels + id] / maxValue;

			if (elements == 2) {

				float hue = atan2f(x, y) / CUDART_PI_F * 0.5f + 0.5f;				
				float value = fminf(sqrtf(x * x + y * y), 1.0f);

				pixels[id] = hsva_to_uint_rgba(hue, 1.0f, value, 1.0f);
			}
			else {
				
				float z = values[2 * numOfPixels + id] / maxValue;

				x = fminf(fmaxf(x, -1), 1);
				y = fminf(fmaxf(y, -1), 1);
				z = fminf(fmaxf(z, -1), 1);

				unsigned char red = (unsigned char) __float2uint_rn(127.5f * (x + 1));
				unsigned char green = (unsigned char) __float2uint_rn(127.5f * (y + 1));
				unsigned char blue = (unsigned char) __float2uint_rn(127.5f * (z + 1));		

				pixels[id] = (0xFF << 24) | (red << 16) | (green << 8) | blue;		
			}
		}
	}
}
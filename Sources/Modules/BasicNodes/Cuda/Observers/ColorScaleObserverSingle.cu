#define _SIZE_T_DEFINED 

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
	__global__ void ColorScaleObserverSingle(float* values, int method, int scale, float minValue, float maxValue, unsigned int* pixels, int numOfPixels)
	{		
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		if(id < numOfPixels) //id of the thread is valid
		{	
			pixels[id] = float_to_uint_rgba(values[id], method, scale, minValue, maxValue);
		}
	}


   /*   Plot memorty block tiles separately next to each other.
	* 
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
	*/
	__device__ int TileObserverGetIdInPixels (int method, int tw, int th, int pw, int pixels_id, int tilesInRow, int noBlocksEffective)
	{
		int ta, values_row, values_col, id_tile, tile_row, tile_col, pixels_row, pixels_col;
		ta           = tw*th;                                       // tile area
		pixels_row   = pixels_id / pw;                              // row-id in the final pixels mem. block (observer)
		pixels_col   = pixels_id % pw;                              // column-id in the final pixels mem. block (observer)
		if (((pixels_col + 1) % (tw + 1) == 0) || ((pixels_row + 1) % (th + 1) == 0)) // it is point between tiles
		{
		   return -1;
		}
		tile_row     = pixels_row / (th+1);                         // in which tile-row it is
		tile_col     = pixels_col / (tw+1);                         // in which tile-column it is
		id_tile      = tile_col + tile_row*tilesInRow;              // which tile it is
		values_row   = pixels_row % (th+1);
		values_col   = pixels_col % (tw+1);
		if (id_tile >= noBlocksEffective) // there can be some empty tiles
		{
			return -1;
		}
		if (method==6) /// RGB
		{
			ta *= 3;  // needs to move three tiles for each tile for RGB :)
		}
		return id_tile*ta + values_col + values_row*tw;   // final oid in the values for red Color:)
	}

	__global__ void ColorScaleObserverTiledSingle(float* values, int method, int scale, float minValue, float maxValue, unsigned int* pixels, int numOfPixels, int pw, int ph, int tw, int th, int tilesInRow, int noBlocksEffective)
	{
		int pixels_id = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		int id;
		if (pixels_id < numOfPixels)
		{
			id = TileObserverGetIdInPixels (method, tw, th, pw, pixels_id, tilesInRow, noBlocksEffective);
			if (id<0) // in the space between tiles
			{
				return;
			}
			pixels[pixels_id] = float_to_uint_rgba(values[id], method, scale, minValue, maxValue);
		}
	}

	__global__ void DrawRGBTiledKernel(float* values, unsigned int* pixels, int pw, int ph, int numOfPixels, int tw, int th, int tilesInRow, int tilesInCol, float maxValue, int noBlocksEffective) 
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	
				+ blockDim.x*blockIdx.x				
				+ threadIdx.x;

		float fred, fgreen, fblue;
		int id_R_value, ta;
		if(id < numOfPixels) //id of the thread is valid
		{
			id_R_value = TileObserverGetIdInPixels (6, tw, th, pw, id, tilesInRow, noBlocksEffective); // nethod == 6 is RGB :)
			ta = tw*th;      
			if (id_R_value<0) // it is point between tiles
			{
				fred = 0.5f;
				fgreen = 0.5f;
				fblue = 0.5f;
			}
			else
			{     
				fred   = values[id_R_value]/maxValue;
				fgreen = values[1 * ta + id_R_value]/maxValue;
				fblue  = values[2 * ta + id_R_value]/maxValue;
			}

			fred   = fminf(fmaxf((fred),   0), 1) * 255;    // normalize colors to be between 0 and 255
			fgreen = fminf(fmaxf((fgreen), 0), 1) * 255;
			fblue  = fminf(fmaxf((fblue),  0), 1) * 255;

			unsigned char red   = (unsigned char) __float2uint_rn(fred);
			unsigned char green = (unsigned char) __float2uint_rn(fgreen);
			unsigned char blue  = (unsigned char) __float2uint_rn(fblue);		

			pixels[id] = (0xFF << 24) | (red << 16) | (green << 8) | blue;		
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

	
	__global__ void DrawRGBKernel(float* values, unsigned int* pixels, int numOfPixels) 
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	
				+ blockDim.x*blockIdx.x				
				+ threadIdx.x;

		if(id < numOfPixels) //id of the thread is valid
		{
			float fred   = values[0 * numOfPixels + id];
			float fgreen = values[1 * numOfPixels + id];
			float fblue  = values[2 * numOfPixels + id];

			fred   = fminf(fmaxf(fred,   0), 1) * 255;
			fgreen = fminf(fmaxf(fgreen, 0), 1) * 255;
			fblue  = fminf(fmaxf(fblue,  0), 1) * 255;

			unsigned char red   = (unsigned char) __float2uint_rn(fred);
			unsigned char green = (unsigned char) __float2uint_rn(fgreen);
			unsigned char blue  = (unsigned char) __float2uint_rn(fblue);		

			pixels[id] = (0xFF << 24) | (red << 16) | (green << 8) | blue;		
		}
	}

	__global__ void DrawGrayscaleKernel(float* values, unsigned int* pixels, int numOfPixels)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x
			+ blockDim.x*blockIdx.x
			+ threadIdx.x;

		if (id < numOfPixels) //id of the thread is valid
		{
			pixels[id] = grayscale_to_uint_rgba(fminf(fmaxf(values[id], 0.0f), 1.0f));
		}
	}

}
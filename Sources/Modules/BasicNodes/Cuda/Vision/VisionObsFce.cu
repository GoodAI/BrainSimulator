//
//
//
// includes for observers.
//
//
//

#define _SIZE_T_DEFINED 

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>
#include "math.h"

#include "../Observers/ColorHelpers.cu"


//#define GET_WHITE ((100 << 24)  | (255 << 16) | (255 << 8) | 255)
//#define GET_RGBA(r,g,b,a) ( ((a) << 24)  | ((r) << 16) | ((g) << 8) | (b) )


extern "C"  
{	





	




	__global__ void FillVBOFromInputImage (float* values, int imageSize, unsigned int* pixels){
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		if (id<imageSize){
            float fval = fminf(fmaxf(values[id], 0), 1);
			int val = fval*255;
			pixels[id] = GET_RGBA(val,val,val,255);
		}
	}

    __global__ void FillVBOHue (float* values, int imageSize, unsigned int* pixels, float hue=0.5f, float sat=0.5f){
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		if (id<imageSize){
            float fval = fminf(fmaxf(values[id], 0), 1);
            pixels[id] = hsva_to_uint_rgba(hue, sat, fval, 1.0f);  // 5.0f is normlaization -> needs to be fixed :(
		}
	}



	__global__ void FillVBO (unsigned int* pixels, int imageSize, int r, int g, int b ){
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		if (id<imageSize){
		pixels[id] = GET_RGBA(r,g,b,255);//GET_WHITE;
		}
	}



    //---- optical flow functions


    __device__ void OFConvertXY2AngleSize (float*of, int id, int imageSize, float& of_size, float& of_angle){
        float2 OF_value;
            
        OF_value.x = of[id];
        OF_value.y = of[id+imageSize];

        of_size  = (float) sqrt( (OF_value.x+OF_value.y) * (OF_value.x+OF_value.y) );  // normalized to be <0,1>
        of_angle = (float) atan2(OF_value.x,OF_value.y);  // <-PI;PI>
    }


    __global__ void OFMapObserver (unsigned int* pixels, int imageSize, float* of){
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

        float OF_size;
        float OF_angle;

		if (id<imageSize){
            OFConvertXY2AngleSize(of,id,imageSize,OF_size,OF_angle);

            OF_size  = OF_size/5.0f; // 5.0f is normlaization -> needs to be fixed :(
            OF_angle = (OF_angle + 3.14159265f)/2.0f; // normalized to be <0,1>

            pixels[id] = hsva_to_uint_rgba(OF_angle, OF_size, 1.0f, 1.0f);  // 5.0f is normlaization -> needs to be fixed :(
		}
	}

    __global__ void OFConvert2AngleSize (float*of, int imageSize){
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

        float OF_size;
        float OF_angle;

		if (id<imageSize){
            OFConvertXY2AngleSize(of,id,imageSize,OF_size,OF_angle);

            of[id] = OF_angle;
            of[id+imageSize] = OF_size;
		}
	}

    __global__ void OFConvert2AngleSizeFloatImage (float*of, int imageSize){
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

        float OF_size;
        float OF_angle;

		if (id<imageSize){
            OFConvertXY2AngleSize(of,id,imageSize,OF_size,OF_angle);

            OF_size  = OF_size/8.0f; // 5.0f is normlaization -> needs to be fixed to proper number... 
            OF_angle = (OF_angle + 3.14159265f)/2.0f; // normalized to be <0,1>

            of[id] = 1-hsva_to_float(OF_angle, OF_size, 1.0f) ;
            of[id+imageSize] = 0.0f;//1-hsva_to_float(OF_angle, OF_size/2.0f, 1.0f);
		}
	}


}


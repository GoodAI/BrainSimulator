//
//
//
// inlcueds for observetes, with soem basic math functions
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

#include "VisionMath.cu"

#include "../Observers/ColorHelpers.cu"





extern "C" 
{

	
	__global__ void Draw_edges(unsigned int* texture , int texture_width , float * adjMat, float * patches , int patches_num , int patches_dim , float * desc , int desc_dim , int type){
		int p  = blockIdx.x; /// patch
		int pc = threadIdx.x; /// its connection
		int r=0,g=0,b=0;

		if (adjMat[p+pc*patches_num]==1){ // if to draw a link
			float2 A = { patches[0 + p * patches_dim] , patches[1 + p * patches_dim] };
			float2 B = { patches[0+pc*patches_dim]    , patches[1+pc*patches_dim] };
			float2 v = { B.x-A.x  ,  B.y-A.y};
			float len = ( (v.x*v.x)+(v.y*v.y) );
			v.x = v.x / len;
			v.y = v.y / len; // normalize

			float dist = Dist_between_two_vec(desc+p*desc_dim , desc+pc*desc_dim , desc_dim);

			for (float i=0; i<len ; i++){
				float2 P = {A.x+v.x*i , A.y+v.y*i};
				if (type==1){ /// if color each edge by the distance
					r = (int)(dist*1000);
			        g = (int)(dist*100);
				    b = (int)(dist*1);
				}
				texture[(int)P.x+(int)P.y*texture_width] = GET_RGBA(r,g,b,255);
			}
		}
	}




	__global__ void FillImWhite(unsigned int* texture , int width , int height){
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (idx<width*height)
			texture[idx]= GET_WHITE;
	}

	



	__device__ int shuffle_id (int idx, int max){
		int new_id = idx/max + (idx%max)*max;
		if (new_id >= max*max)
			new_id =max*max;
		return new_id-1;
	}



	__global__  void FillImByOtherIm (unsigned int * texture , float* im_source , int size , int max_segs){
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (idx<size){
			float seg_color_value = (float)shuffle_id(im_source[idx],max_segs) /max_segs/max_segs ;
			//float seg_color_value = im_source[shuffle_id(idx,max_segs)]/max_segs/max_segs;
			texture[idx] = hsva_to_uint_rgba(seg_color_value, 1.0, 1.0, 1);
		}
	}
	__global__  void FillImByOtherImWithMovent (unsigned int * texture , int texWidth, int texHight, float* im_source , int imWidth, int imHight, int imCornerInTextureX, int imCornerInTextureY)
	{
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		int idx_x, idx_y, x, y, idx_tex, col;
		if (idx<imWidth*imHight){
			idx_x = idx%imWidth;
			idx_y = idx/imWidth;
			x = imCornerInTextureX + idx_x;
			y = imCornerInTextureY + idx_y;
			idx_tex = x + y*texWidth;
			if (idx_tex<texWidth*texHight){
				col = (int)(255*im_source[idx]);
				texture[idx_tex] = GET_RGBA(col,col,col,1);
				//texture[idx_tex] = im_source[idx];
			}
		}
	}






	__global__  void FillImByEnergy (unsigned int * texture , float* mask , int size_im , float* energiesPerElement , float max_energy, float min_energy){
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (idx<size_im){
			int id_element = mask[idx];
			float energy   = energiesPerElement[id_element];
            float value    = max_energy!=0  ?  (energy-min_energy)/max_energy  :  (energy-min_energy);  
			texture[idx]   = hsva_to_uint_rgba(1.0f, 0.0f, value, 1.0f);
		}
	}



	





}



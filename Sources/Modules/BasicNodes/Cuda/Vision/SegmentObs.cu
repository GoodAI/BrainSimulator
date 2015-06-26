#define _SIZE_T_DEFINED 

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>


#include "SegmentDefs.cu"
#include "../Observers/ColorHelpers.cu"



extern "C" 
{

	
	__global__ void Draw(unsigned int* texture , SLICClusterCenter* vSLICCenterList , int* maskBuffer , int width , int height ,  int SLIClist_size , int type){
		int mskIndex = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		float4 lab;
		int centerIndex;
		if (mskIndex<width*height){
			centerIndex = maskBuffer[mskIndex];
			switch (type){
			case 0:    //--- marked
				if (centerIndex!=maskBuffer[mskIndex+1]  ||  centerIndex!=maskBuffer[mskIndex-width]){
					texture[mskIndex]= GET_WHITE;
				}
				else{
					lab = vSLICCenterList[centerIndex].lab;
					texture[mskIndex]=0;
				}
				break;
			case 1:   //--- average color...
				if (centerIndex>=0){
					lab = vSLICCenterList[centerIndex].lab;
					lab.x *=255; /// IM not sure about this niether...
					lab.y *=255;
					lab.z *=255;
					int r = 0.41847 * lab.x + (-0.15866) * lab.y + (-0.082835) * lab.z;
					int g =(-0.091169) * lab.x + 0.25243 * lab.y + 0.0015708 * lab.z;
					int b = 0.00092090 * lab.x + (-0.0025498) * lab.y + 0.17860 * lab.z;
					texture[mskIndex] = GET_RGBA(r*100,g*100,b*100,255); /// This norm is not correct
				}
				break;
			case 2: //--- copu mask buffer (matrix) to texture
				texture[mskIndex] = GET_RGBA(centerIndex,centerIndex,centerIndex,100);
				break;
			case 3: //--- copu mask buffer (matrix) to texture // shuffle
				int centerIndex_shf = (  ((float)(centerIndex%15)/15) *255  );
				texture[mskIndex] = GET_RGBA(centerIndex_shf,centerIndex_shf,centerIndex_shf,100);
				break;
			case 4: //--- only update borders, nothing else...
				if (centerIndex!=maskBuffer[mskIndex+1]  ||  centerIndex!=maskBuffer[mskIndex-width]){
					texture[mskIndex]= 0;
				}
				break;
			}
		}
	}


	__global__ void Draw_centers(unsigned int* texture , SLICClusterCenter* vSLICCenterList , int* maskBuffer , int width , int height ,  int SLIClist_size , int type){
		int centerIdx = blockDim.x;
		switch (type){
		case 0:    //--- mark centers
				float2 xy = vSLICCenterList[centerIdx].xy;
				int x = xy.x;
				int y = xy.y;
				for (int i=-1 ; i<=1 ; i++)
					for (int j=-1 ; j<=1 ; j++)
						if (x>0 && y>0 && y<height-1 && x<width-1)
							texture[x+i+(y+j)*width] = GET_RGBA(255,100,100,100);
				texture[x+y*width] = GET_RGBA(255,0,0,100);

			}
	
	}


	__global__ void Test_draw_xy(unsigned int* texture , int texture_width , float * xy, int xy_dim, int xy_length){
		for (int p=0 ; p<xy_length ; p++){
			int x = xy[0 + p*xy_dim];
			int y = xy[1 + p*xy_dim];

			for (int i=-1 ; i<=1 ; i++)
					for (int j=-1 ; j<=1 ; j++)
						texture[x+i+(y+j)*texture_width] = GET_RGBA(0,100,0,0);
			texture[x+y*texture_width] = GET_RGBA(100,255,100,0);
						
		}


	}





}
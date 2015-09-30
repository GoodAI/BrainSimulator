#define _SIZE_T_DEFINED 
/*
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif
*/

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>
#include <math.h>



#include "../Observers/ColorHelpers.cu"
#include "VisionMath.cu"





extern "C" 
{

	/// add new descriptor as completly new cluster center
	__global__ void AddDataAsCC(		
		float *CC,
		int desc_dim, /// dimesion of desc
		int id_CC,    /// id where to write
		float *desc2add,
		float *N_in_CC
		){
			int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x	+ threadIdx.x;
			if(id < desc_dim){
				CC[id_CC*desc_dim + id] = desc2add[id];
			}
			if (id==0) /// this nedes to be done just once, so first worker will do that
				++N_in_CC[id_CC];
	}




		/// add new descriptor as completly new cluster center
	__global__ void UpdateCC_XY(		
		float *CCXY,
		int id_CC,
		float *XY_tofill,
		int dim_XY
		){
			int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x	+ threadIdx.x;
			if(id < dim_XY)
				CCXY[id_CC*dim_XY + id] = XY_tofill[id];
	}










	/// move desc of CC to the new one as kind of average :)
	__global__ void UpadateCC_Desc(
		float *CC,
		int desc_dim,    /// dimesion of desc
		int id_CC,       /// whcih CC to move
		float *desc2add,
		float *N_in_CC,
		float movingKMeans_alpha
		){
			int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x	+ threadIdx.x;
			float last_value;

			//--- move the value
			if(id < desc_dim){
				last_value = CC[id_CC*desc_dim + id];
				CC[id_CC*desc_dim + id] = (1-movingKMeans_alpha)*last_value + desc2add[id]*movingKMeans_alpha;   ///   move XY to the average
			}

			__syncthreads;
			//--- increase var where # of elements for each cluster was stored...
			if (id==0)
				++N_in_CC[id_CC];
	}


		/// add new descriptor as completly new cluster center
	__global__ void ApplyBrainsMovement(
		float *CCXY,
		int dim_XY,
		float *movement,
		int dim_movement,
		int max_clusters
		){
			int id = blockDim.x*blockIdx.y*gridDim.x   + blockDim.x*blockIdx.x   + threadIdx.x;
			if (id<max_clusters){
				//--- move in XY
				if (dim_movement>=2){
					CCXY[id*dim_XY]   -= movement[0];
					CCXY[id*dim_XY+1] -= movement[1];
				}
				//--- apply rotation in X
				if (dim_movement>=3){
				}
			}
	}




	//
	// ============================================================================
	//   R E S O R T I N G
	// ----------------------------------------------------------------------------
	//
	__global__  void Copy_A_to_B (float * A , float * B , int size){
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<size)
			B[id] = A[id];
	}
	__global__  void Copy_matA_to_matB_withShuffleIdx (float * A , float * B , int size, int cols , float * new_idxs, int max_rows){
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		int irow = id / cols;
		int icol = id % cols;
		if (id<size){
			int irow_new = max_rows - 1 - irow; /// it was ascending, so I need to revert it...
			int irow_old = new_idxs[irow];
			B[irow_new*cols + icol] = A[irow_old*cols + icol];
		}
	}







	//
	// ============================================================================
	//   O B S E R V E R
	// ----------------------------------------------------------------------------
	//
	__global__  void FillImByActState (unsigned int * texture , int dim_texture, int size_texture , float * XY, int dim_xy, int N_objcets , int weightByN , float * NClusters, float maxClusFreq, int isXYinOneNorm){
		int id_Object = blockIdx.x;
		int id_Plot_i = threadIdx.x;
		int id_Plot_j = threadIdx.y;
		//int id_xy = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;

		float2 xy;
		xy.x = XY[id_Object*dim_xy];
		xy.y = XY[id_Object*dim_xy+1];

		if (isXYinOneNorm)
		{
			xy.x = (xy.x+1)*250;
			xy.y = (xy.y+1)*150;
		}

		float colValue = 1.0f;

		int id_tex_draw = ((int)xy.y+id_Plot_i)*dim_texture+((int)xy.x+id_Plot_j);
		if (id_tex_draw<size_texture)
			if (weightByN==1)
				colValue = NClusters[id_Object]/maxClusFreq;
			colValue=colValue>1.0f?1.0f:colValue;
			texture[id_tex_draw]   = hsva_to_uint_rgba((float)id_Object/(float)N_objcets, 1.0f,colValue,1.0f);
		}




	__global__  void DownInTime (unsigned int * texture , int size_im){
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (idx<size_im){
/*			int r = (char)(texture[idx] >> 16);//((100 << 24)  | (255 << 16) | (255 << 8) | 255)
			int g = (char)(texture[idx] >> 8);//((100 << 24)  | (255 << 16) | (255 << 8) | 255)
			int b = (char)(texture[idx] >> 0);//((100 << 24)  | (255 << 16) | (255 << 8) | 255)
*/			/*r = 255-(255-r)*0.5;
			g = 255-(255-g)*0.5;
			b = 255-(255-b)*0.5;*/
			texture[idx]   = GET_RGBA(0,0,0,255);
		}
	}

	__global__  void Test (unsigned int * texture , int size_im){
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		int max_energy = 500;
		if (idx<size_im)
			texture[idx]   = hsva_to_uint_rgba((idx%max_energy)/(float)max_energy, 1.0f,1.0f,1.0f);
	}









	__device__ void deviceFocuserInputObserver(int id, float* values, float cX, float cY, float subImgDiameter, int inputWidth, int inputHeight, unsigned int* pixels, float color, int keepLines=1)
	{
    		int maxDiameter = min(inputWidth, inputHeight);
			int diameterPix = (int)(subImgDiameter * maxDiameter);
			diameterPix = max(1, diameterPix);

			int cXPix = (int)(inputWidth * (cX + 1) * 0.5f);
			int cYPix = (int)(inputHeight * (cY + 1) * 0.5f);

			int subImgX = cXPix - diameterPix / 2;
			int subImgY = cYPix - diameterPix / 2;

			subImgX = max(subImgX, 0);
			subImgY = max(subImgY, 0);

			subImgX = min(subImgX, inputWidth - diameterPix);
			subImgY = min(subImgY, inputHeight - diameterPix);

			int px = id % inputWidth;
			int py = id / inputWidth;			

			float hue = color;//1.0f;
			float saturation = 0;				
			float value = values[id];			

			//--- color of rectange
			if (px >= subImgX && py >= subImgY      && 		px <= subImgX + diameterPix && py <= subImgY + diameterPix) 
			{
				saturation = 0.33f;
				value += 0.2f;
			}	

			//--- cross
			if (px == cXPix || py == cYPix) 
			{
				saturation = 0.5f;
				value += 0.7f;
			}			

			//--- update what is inside box
			//if (  (px >= subImgX && py >= subImgY &&  px <= subImgX + diameterPix && py <= subImgY + diameterPix) ) // update only inside box :)
            if (  (keepLines && (cXPix == px || cYPix == py)) || (px >= subImgX && py >= subImgY &&  px <= subImgX + diameterPix && py <= subImgY + diameterPix)  ) // update insde box and keep lines around
            {
				value = fminf(fmaxf(value, 0), 1);
				pixels[id] = hsva_to_uint_rgba(hue, saturation, value, 1.0f);
			}
	}




	__global__ void FocuserInputObserver(float* values, float* pupilControl, int inputWidth, int inputHeight, unsigned int* pixels, float color, int keepLines=1)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		int numOfPixels = inputWidth * inputHeight;

		if(id < numOfPixels) //id of the thread is valid
		{		
            float cX = pupilControl[0]; // <-1, 1>
			float cY = pupilControl[1]; // <-1, 1>
			float subImgDiameter = pupilControl[2]; // <0,1>

            deviceFocuserInputObserver(id, values, cX, cY, subImgDiameter, inputWidth, inputHeight, pixels, color,keepLines);
		}
	}



    /// input location is defined as center + location :)

	__global__ void FocuserInputObserver_withMovementDirection(float* values, float* pupilControl_direction, float* pupilControl_center, int id_pupil , int inputWidth, int inputHeight, unsigned int* pixels, float color)
	{

		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

//        float pupilControl[3];
		int numOfPixels = inputWidth * inputHeight;

		if(id < numOfPixels) //id of the thread is valid
		{		
            float cX = pupilControl_direction[0]+pupilControl_center[0];
			float cY = pupilControl_direction[1]+pupilControl_center[1];
			float subImgDiameter = pupilControl_direction[2]+pupilControl_center[2];
            deviceFocuserInputObserver(id, values, cX, cY, subImgDiameter, inputWidth, inputHeight, pixels, color);
		}
	}





}





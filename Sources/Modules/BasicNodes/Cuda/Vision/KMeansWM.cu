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
		int max_ = 500;

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
			int r = (char)(texture[idx] >> 16);//((100 << 24)  | (255 << 16) | (255 << 8) | 255)
			int g = (char)(texture[idx] >> 8);//((100 << 24)  | (255 << 16) | (255 << 8) | 255)
			int b = (char)(texture[idx] >> 0);//((100 << 24)  | (255 << 16) | (255 << 8) | 255)
			/*r = 255-(255-r)*0.5;
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









	__device__ void deviceFocuserInputObserver(int id, float* values, float cX, float cY, float subImgDiameter, int inputWidth, int inputHeight, unsigned int* pixels, float color)
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
			if (  (px >= subImgX && py >= subImgY &&  px <= subImgX + diameterPix && py <= subImgY + diameterPix) )
            {
				value = fminf(fmaxf(value, 0), 1);
				pixels[id] = hsva_to_uint_rgba(hue, saturation, value, 1.0f);
			}
	}




	__global__ void FocuserInputObserver(float* values, float* pupilControl, int id_pupil , int inputWidth, int inputHeight, unsigned int* pixels, float color)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

		int numOfPixels = inputWidth * inputHeight;

		if(id < numOfPixels) //id of the thread is valid
		{		
			float cX = pupilControl[id_pupil*3 + 0]; // <-1, 1>
			float cY = pupilControl[id_pupil*3 + 1]; // <-1, 1>
			float subImgDiameter = pupilControl[id_pupil*3 + 2]; // <0,1>

            deviceFocuserInputObserver(id, values, cX, cY, subImgDiameter, inputWidth, inputHeight, pixels, color);
		}
	}



    /// input location is defined as center + location :)

	__global__ void FocuserInputObserver_withMovementDirection(float* values, float* pupilControl_direction, float* pupilControl_center, int id_pupil , int inputWidth, int inputHeight, unsigned int* pixels, float color)
	{

		int id = blockDim.x*blockIdx.y*gridDim.x	
			+ blockDim.x*blockIdx.x				
			+ threadIdx.x;

        float pupilControl[3];
		int numOfPixels = inputWidth * inputHeight;

		if(id < numOfPixels) //id of the thread is valid
		{		
            float cX = pupilControl_direction[0]+pupilControl_center[0];
			float cY = pupilControl_direction[1]+pupilControl_center[1];
			float subImgDiameter = pupilControl_direction[2]+pupilControl_center[2];
            deviceFocuserInputObserver(id, values, cX, cY, subImgDiameter, inputWidth, inputHeight, pixels, color);
		}
	}




	//
	//
	//============================================================================================================
	//============================================================================================================
	//
	//       - - -   S P A T I  A L      R E L A T I O N S  - - -
	//
	//------------------------------------------------------------------------------------------------------------
	//------------------------------------------------------------------------------------------------------------
	//
	//




	__global__  void W_update (float * W, int n_W, const float * obj_sim1, const float * obj_sim2, int n_objs, const float * spat_rel , int n_spat_rel)
	{
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;

		int i_spat_rel = idx / (n_objs * n_objs); 
		int i_obj1 = (idx % (n_objs * n_objs)) / n_objs;
		int i_obj2 = idx%n_objs; //idx - i_obj1 * n_objs - i_spat_rel * n_objs * n_objs;

		if (idx<n_W){
			float val = obj_sim1[i_obj1] * obj_sim2[i_obj2] * spat_rel[i_spat_rel];
			W[idx] += val;
		}
	}









		__device__ void Draw_edge_dev(unsigned int* texture , int texture_width , float * A, float * B , float color=0.f)
		{
			float2 v = { B[0]-A[0]  ,  B[1]-A[1]};
			float len = ( (v.x*v.x)+(v.y*v.y) );
			v.x = v.x / len;
			v.y = v.y / len; // normalize
			int r=0,g=0,b=0;

			for (float i=0; i<len ; i++)
			{
				float2 P = {A[0]+v.x*i , A[1]+v.y*i};
					r = (int)(color*1000);
			        g = (int)(color*100);
				    b = (int)(color*1);
					int tex_index = (int)P.x+(int)P.y*texture_width;
					if (tex_index>0 && tex_index<texture_width*texture_width)
					{
						texture[(int)P.x+(int)P.y*texture_width] = GET_RGBA(r,g,b,255);
					}
			}
		}

		__global__ void Draw_edge_glob(unsigned int* texture , int textureWidth , int textureHeight, float Ax, float Ay, float Bx, float By, int max_edge_len, float color)
		{
			int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;

			Ax = textureWidth * (Ax + 1) * 0.5f;
			Ay = textureHeight * (Ay + 1) * 0.5f;
			Bx = textureWidth * (Bx + 1) * 0.5f;
			By = textureHeight * (By + 1) * 0.5f;

			float2 v = { Bx-Ax  ,  By-Ay};
			float len = sqrt( (v.x*v.x)+(v.y*v.y) );
			v.x = v.x / len;
			v.y = v.y / len; // normalize
			//int r=(int)(color*1000), g=(int)(color*100), b=(int)(color*1);

			if (idx<len && idx<max_edge_len){
				float2 P = {Ax+v.x*idx , Ay+v.y*idx};
				int tex_index = (int)P.x+(int)P.y*textureWidth;
				if (tex_index>0 && tex_index<textureWidth*textureHeight)
				{
					texture[tex_index] = hsva_to_uint_rgba(color,0.9f,0.9f,1.0f);
					//texture[(int)P.x+(int)P.y*textureWidth] = GET_RGBA(r,g,b,255);
				}
			}
		}
	__global__ void Draw_edge_globXY(unsigned int* texture , int textureWidth , int textureHeight, float Ax, float Ay, float Bx, float By, int max_edge_len, float color)
		{
			int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;

			float2 v = { Bx-Ax  ,  By-Ay};
			float len = sqrt( (v.x*v.x)+(v.y*v.y) );
			v.x = v.x / len;
			v.y = v.y / len; // normalize
			//int r=(int)(color*1000), g=(int)(color*100), b=(int)(color*1);

			if (idx<len && idx<max_edge_len){
				float2 P = {Ax+v.x*idx , Ay+v.y*idx};
				int tex_index = (int)P.x+(int)P.y*textureWidth;
				if (tex_index>0 && tex_index<textureWidth*textureHeight)
				{
					texture[tex_index] = hsva_to_uint_rgba(color,0.9f,0.9f,1.0f);
					//texture[(int)P.x+(int)P.y*textureWidth] = GET_RGBA(r,g,b,255);
				}
			}
		}


		__global__ void Draw_point_glob(unsigned int* texture , int textureWidth , int textureHeight, float x, float y, float color, int point_width)
		{
			int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;

			if (idx<point_width*point_width){
				//--- convert to x,y
				x = textureWidth * (x + 1) * 0.5f;
				y = textureHeight * (y + 1) * 0.5f;

				//--- calculate neigh
				x += idx/point_width-1;
				y += idx%point_width-1;
				int tex_index = (int)x+(int)y*textureWidth;

				if (tex_index>0 && tex_index<textureWidth*textureHeight)
				{
					texture[tex_index] = hsva_to_uint_rgba(color,0.9f,0.9f,1.0f);
				}
			}
		}

	__global__ void Draw_point_globXY(unsigned int* texture , int textureWidth , int textureHeight, float x, float y, float color, int point_width)
		{
			int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;

			if (idx<point_width*point_width){
    			//--- calculate neigh
				x += idx/point_width-1;
				y += idx%point_width-1;
				int tex_index = (int)x+(int)y*textureWidth;

				if (tex_index>0 && tex_index<textureWidth*textureHeight)
				{
					texture[tex_index] = hsva_to_uint_rgba(color,0.9f,0.9f,1.0f);
				}
			}
		}




	__device__ float2 get_min_max_of_vec(const float * vec, int size , int icx=1){
		float2 min_max;
		min_max.x = FLT_MAX;
		min_max.y = -FLT_MAX;
		for (int i=0; i<size; i+=icx){
			if (vec[i]<min_max.x){
				min_max.x = vec[i];
			}
			if (vec[i]>min_max.y){
				min_max.y = vec[i];
			}
		}
		return min_max;
	}



	__device__ void plot_image_patch(unsigned int* texture, int textureWidth, float weight, float * image_patch, int image_patch_size, int location_index, float probabl, float h_color = .5f)
	{
		float hue = h_color;
		float saturation = .7f;				
		float value = .8f;weight;			

		int patch_dim = 11; /// image dimensions, sqrt(image_patch_size);
		int dim_extend = 1;
		
		float2 min_max = get_min_max_of_vec(image_patch,image_patch_size);
		
		int a=1;

		for (int row=-patch_dim/2; row<patch_dim/2+1 ; row++){
			for (int col=-patch_dim/2; col<patch_dim/2+1 ; col++){
				int patch_index = (col+patch_dim/2)*patch_dim+row+patch_dim/2;
				for (int ir = 0 ; ir<dim_extend ; ir++){
					for (int ic = 0 ; ic<dim_extend ; ic++){
						int texture_index = location_index   +   dim_extend*row + ir    +   (dim_extend*col+ic)*textureWidth;
						if (texture_index>0 && texture_index < textureWidth*textureWidth-1)
						{
							texture[texture_index] = hsva_to_uint_rgba(hue, .0f, ((float)row+patch_dim/2)/patch_dim, 1.0f);
							if (  (row+ir)==(-patch_dim/2+0) || (row+ir)==(patch_dim/2+dim_extend-1) || (col+ic)==(-patch_dim/2+0)  || (col+ic)==(patch_dim/2+dim_extend-1)  )
							{
								texture[texture_index] = hsva_to_uint_rgba(hue, saturation, value, 1.0f);
							}
							else
							{
								texture[texture_index] = hsva_to_uint_rgba(hue, .0f, (image_patch[patch_index]-min_max.x)/(min_max.y-min_max.x), 1.0f);
								texture[texture_index] = hsva_to_uint_rgba(hue, (1-probabl), (image_patch[patch_index]-min_max.x)/(min_max.y-min_max.x), 1.0f);
							}
						}
					}	
				}
			}
		}
		//for (int i=0; i<40; i++) texture[location_index] = hsva_to_uint_rgba(hue, .0f, 0.5, 1.0f);
	}


	__global__ void Plot_Ws_guess(unsigned int* texture, int textureWidth, int textureHeight, float* current_obj_location, int seen_object_id, float * objs, int dim_objs, float* W, int n_W, int n_objs_in_dbse, float * spat, int n_spat, int dim_spat)
	{
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		int i_spat_rel = idx / (n_objs_in_dbse * n_objs_in_dbse); 
		int i_obj1 = (idx % (n_objs_in_dbse * n_objs_in_dbse)) / n_objs_in_dbse;
		int i_obj2 = idx%n_objs_in_dbse;// - i_obj1 * n_objs_in_dbse - i_spat_rel * n_objs_in_dbse * n_objs_in_dbse;

		float *image_patch = objs;
		int guessed_location_index;

		float cX = current_obj_location[0]; 
		float cY = current_obj_location[1]; 

		int cXPix = (int)(textureWidth * (cX + 1) * 0.5f);
		int cYPix = (int)(textureHeight * (cY + 1) * 0.5f);

		if (idx<n_W){
			if (i_obj1==seen_object_id){
				float2 W_min_max = get_min_max_of_vec(W+seen_object_id,n_W,n_objs_in_dbse);
				if (W[idx] > W_min_max.y*0.8){
					image_patch = objs + i_obj2*dim_objs;
					float spat_rel_vecX = (float)textureWidth*(spat[i_spat_rel*dim_spat+0])*.5f;
					float spat_rel_vecY = (float)textureHeight*(spat[i_spat_rel*dim_spat+1])*.5f;
					float B[2]; B[0] = (float)cXPix + spat_rel_vecX; B[1] = (float)cYPix+spat_rel_vecY;
					float P1 = spat[i_spat_rel*dim_spat+0];
					float P2 = spat[i_spat_rel*dim_spat+1];
					guessed_location_index = (int)B[0] + ((int)B[1])*textureWidth;
					float prb = W[idx]/W_min_max.y;
					plot_image_patch(texture,textureWidth,W[idx],image_patch,dim_objs,guessed_location_index, prb*prb*prb);

					float A[2]; A[0] = cXPix; A[1] = cYPix;
					Draw_edge_dev(texture,textureWidth,A,B,0.2);
				}  
			}
		}
	}

	__global__ void Test_Ws(float* W, int n_W, int n_objs, float * out)
	{
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		int i_spat_rel = idx / (n_objs * n_objs); 
		int i_obj1 = (idx % (n_objs * n_objs)) / n_objs;
		int i_obj2 = idx - i_obj1 * n_objs - i_spat_rel * n_objs * n_objs;

		if (idx<n_W){
			if (i_obj1==2){
				atomicAdd(out,1);
				//out[0] += 1;
			}
		}

	}



	

	//
	//
	//============================================================================================================
	//============================================================================================================
	//
	//       - - -   S P A T I A L      R E L A T I O N S   !!! R B M !!    - - -
	//
	//------------------------------------------------------------------------------------------------------------
	//------------------------------------------------------------------------------------------------------------
	//
	//
	
	
	__global__ void Plot_RBM_guess_basic(
		unsigned int* texture, int textureWidth, int textureHeight,
		float* current_obj_location,
		int o0_id,
		int o1_id,
		int spat_id,
		float alpha,
		float * objs, int n_objs, int dim_objs,
		float * spat, int n_spat, int dim_spat)
	{
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		//int i_spat_rel = idx / (n_objs_in_dbse * n_objs_in_dbse); 
	//	int i_obj1 = (idx % (n_objs_in_dbse * n_objs_in_dbse)) / n_objs_in_dbse;
		//int i_obj2 = idx%n_objs_in_dbse;// - i_obj1 * n_objs_in_dbse - i_spat_rel * n_objs_in_dbse * n_objs_in_dbse;

		float *image_patch = objs;
		int guessed_location_index;

		float cX = current_obj_location[3]; 
		float cY = current_obj_location[4]; 

		int cXPix = (int)(textureWidth * (cX + 1) * 0.5f);
		int cYPix = (int)(textureHeight * (cY + 1) * 0.5f);

		//if (idx<n_W){
		//	if (i_obj1==seen_object_id){
				//float2 W_min_max = get_min_max_of_vec(W+seen_object_id,n_W,n_objs_in_dbse);
				//if (W[idx] > W_min_max.y*0.8){
					image_patch = objs + o1_id*dim_objs;
					float spat_rel_vecX = (float)textureWidth*(spat[spat_id*dim_spat+0])*.5f;
					float spat_rel_vecY = (float)textureHeight*(spat[spat_id*dim_spat+1])*.5f;
					float B[2]; B[0] = (float)cXPix + spat_rel_vecX; B[1] = (float)cYPix+spat_rel_vecY;
					float P1 = spat[spat_id*dim_spat+0];
					float P2 = spat[spat_id*dim_spat+1];
					guessed_location_index = (int)B[0] + ((int)B[1])*textureWidth;
					float prb = 0.5;//W[idx]/W_min_max.y;
					plot_image_patch(texture,textureWidth,textureHeight,image_patch,dim_objs,guessed_location_index, alpha);

					float A[2]; A[0] = cXPix; A[1] = cYPix;
					Draw_edge_dev(texture,textureWidth,A,B,0.2);
			//	}  
			//}
		//}
	}

	__global__ void Plot_Obj_at_postion(
		unsigned int* texture, int textureWidth, int textureHeight,
		float cX, float cY,
		int o_id,
		float alpha, float h_color,
		float * objs, int n_objs, int dim_objs)
	{
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
	
		if (idx==0){
			float *image_patch = objs;
			int guessed_location_index;

			//float cX = 0.0f;//current_obj_location[0]; 
			//float cY = current_obj_location[1]; 

			int cXPix = (int)(textureWidth * (cX + 1) * 0.5f);
			int cYPix = (int)(textureHeight * (cY + 1) * 0.5f);

			image_patch = objs + o_id*dim_objs;
			guessed_location_index = (int)cXPix + ((int)cYPix)*textureWidth;
			plot_image_patch(texture,textureWidth,textureHeight,image_patch,dim_objs,guessed_location_index, alpha, h_color);
		}
	}

	__global__ void Plot_Input_at_postion(
		unsigned int* texture, int textureWidth, int textureHeight,
		float cX, float cY,
		float alpha, float h_color,
		float * obj, int dim_obj)
	{
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
	
		if (idx==0){
			int cXPix = (int)(textureWidth * (cX + 1) * 0.5f);
			int cYPix = (int)(textureHeight * (cY + 1) * 0.5f);
			int guessed_location_index = (int)cXPix + ((int)cYPix)*textureWidth;
			plot_image_patch(texture,textureWidth,textureHeight,obj,dim_obj,guessed_location_index, alpha, h_color);
		}
	}
__global__ void Plot_Input_at_postionXY(
		unsigned int* texture, int textureWidth, int textureHeight,
		float cX, float cY,
		float alpha, float h_color,
		float * obj, int dim_obj)
	{
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
	
		if (idx==0){
			int guessed_location_index = (int)cX + ((int)cY)*textureWidth;
			plot_image_patch(texture,textureWidth,textureHeight,obj,dim_obj,guessed_location_index, alpha, h_color);
		}
	}


	/*
	__global__ void Plot_RBM_guess(
		unsigned int* texture, int textureWidth, int textureHeight,
		float* current_obj_location,
		float* estimated_obj0, int n_objs_in_dbse,
		float* estimated_obj1,
		float* estimated_spat, int n_spats_in_dbse,
		float * objs, int n_objs, int dim_objs,
		float * spat, int n_spat, int dim_spat)
	{
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		int i_spat_rel = idx / (n_objs_in_dbse * n_objs_in_dbse); 
		int i_obj1 = (idx % (n_objs_in_dbse * n_objs_in_dbse)) / n_objs_in_dbse;
		int i_obj2 = idx%n_objs_in_dbse;// - i_obj1 * n_objs_in_dbse - i_spat_rel * n_objs_in_dbse * n_objs_in_dbse;

		float *image_patch = objs;
		int guessed_location_index;

		float cX = current_obj_location[0]; 
		float cY = current_obj_location[1]; 

		int cXPix = (int)(textureWidth * (cX + 1) * 0.5f);
		int cYPix = (int)(textureHeight * (cY + 1) * 0.5f);

		if (idx<n_W){
			if (i_obj1==seen_object_id){
				float2 W_min_max = get_min_max_of_vec(W+seen_object_id,n_W,n_objs_in_dbse);
				if (W[idx] > W_min_max.y*0.8){
					image_patch = objs + i_obj2*dim_objs;
					float spat_rel_vecX = (float)textureWidth*(spat[i_spat_rel*dim_spat+0])*.5f;
					float spat_rel_vecY = (float)textureHeight*(spat[i_spat_rel*dim_spat+1])*.5f;
					float B[2]; B[0] = (float)cXPix + spat_rel_vecX; B[1] = (float)cYPix+spat_rel_vecY;
					float P1 = spat[i_spat_rel*dim_spat+0];
					float P2 = spat[i_spat_rel*dim_spat+1];
					guessed_location_index = (int)B[0] + ((int)B[1])*textureWidth;
					float prb = W[idx]/W_min_max.y;
					plot_image_patch(texture,textureWidth,W[idx],image_patch,dim_objs,guessed_location_index, prb*prb*prb);

					float A[2]; A[0] = cXPix; A[1] = cYPix;
					Draw_edge(texture,textureWidth,A,B,0.2);
				}  
			}
		}
	}

	*/

}





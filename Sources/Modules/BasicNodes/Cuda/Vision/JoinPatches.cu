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







extern "C" 
{


	// check for each neigh pixels whether it is a border and if so, save it into a adjacency matrix
	__global__ void FillAdjacencyMatrix(float* adj_mat , float* maskBuffer , int size , int cols , int rows ,int Nsegs){
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		int icol = idx % cols;
		int irow = idx / cols;
		int seg_id1=-1;
		if (idx<size){
			if (icol<cols-2 && irow<rows-2 && irow>1 && icol>1){
				seg_id1 = maskBuffer[idx];				
				if (seg_id1!=maskBuffer[idx+1]){
					adj_mat[ (int)maskBuffer[idx+1] + seg_id1*Nsegs ]=1;
					adj_mat[ seg_id1 + Nsegs*(int)maskBuffer[idx+1] ]=1; /// it can happen that a->b, but b->a wont appear...
				}
				else if (seg_id1!=maskBuffer[idx-cols]){
					adj_mat[ (int)maskBuffer[idx-cols] + seg_id1*Nsegs ]=1;
					adj_mat[ seg_id1 + Nsegs*(int)maskBuffer[idx-cols] ]=1; /// it can happen that a->b, but b->a wont appear...
				}
			}
		}
	}


	

	// TODO: do with TILES
	//    and atomicadd precompute for all share_lab
	__global__ void CumulatePositionOfNewObjects(float* mask , float* maskNewIds , float* maskOut, int mask_size, int mask_cols, float* centers, int centers_size, int centers_columns){
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		int icol = idx % mask_cols;
		int irow = idx / mask_cols;

		int i_mask, i_obj;

		if (idx<mask_size){
			i_mask = mask[idx];
			i_obj  = maskNewIds[i_mask];
			maskOut[idx] = i_obj; 
			if (i_obj*centers_columns+2<centers_size){
				atomicAdd(centers + 0 + i_obj*centers_columns , (float)icol);
				atomicAdd(centers + 1 + i_obj*centers_columns , (float)irow);
				atomicAdd(centers + 2 + i_obj*centers_columns , 1.0f);
			}
		}
	}


}





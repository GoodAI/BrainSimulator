

/*
*
*   To work with matrixes.
*
*/


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

#include "math.h"




#include <limits.h>




extern "C" 
{


	__global__  void ResetImage (float* im , int size)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<size)
			im[id] = 0;
	}

	__global__  void SetValue (float* im , float val, int size)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<size)
			im[id] = val;
	}


	__global__  void Multiply(float* im, float val, int size)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;
		if (id<size)
			im[id] *= val;
	}


	// fil vector between idmin and idmadx to value :)
	__global__ void SetVauleInIdxMinMax(
		float* vector, 
		int id_min, 
		int id_max, 
		float value)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id>=id_min && id<=id_max)
				vector[id] = value;
	}


    __global__ void SetMatrixVauleMinMaxX(
		float* matrix, 
        int cols,
        int size,
		int id_min, 
		int id_max, 
		float value)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
        int id_column = id%cols;
		if (id_column>=id_min && id_column<=id_max && id<size)
				matrix[id] = value;
	}
    __global__ void SetMatrixVauleMinMaxY(
		float* matrix, 
        int cols,
        int size,
		int id_min, 
		int id_max, 
		float value)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
        int id_row = id/cols;
		if (id_row>=id_min && id_row<=id_max && id<size)
				matrix[id] = value;
	}


	__global__  void MatrixCopy (float* in , float *out, int size)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<size)
			out[id] = in[id];
	}


		// this is called from kernel, so no idx... 
	__device__ float Dist_between_two_vec(float * v0 , float *v1 , int size){
			float dist = 0;
			for (int i=0 ; i<size ; i++)
				dist+= (v0[i]-v1[i])*(v0[i]-v1[i]);
			
			return sqrt(dist);
	}
	__global__ void Dist_between_two_vec_naive(float * v0 , float *v1 , int size , float * dst){
			float dist = 0;
			for (int i=0 ; i<size ; i++)
				dist+= (v0[i]-v1[i]);//*(v0[i]-v1[i]);
			
			dst[0] = dist;
	}


	
	////////////////////////////////////////////////////////////////////////
	////////////////////////////////////////////////////////////////////////
	//
	//
	// Stuff with matrices...
	//
	//
	////////////////////////////////////////////////////////////////////////
	
	
	// A+B=C
	__global__ void Sum(float * A , float  *B , float *C, int size) {
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<size){
			C[id] = A[id]+B[id];
		}
	}

	__global__ void Round(float * A , float  *out, int size) {
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<size){
			int t   =  (int)(out[id]+0.5);  // can it be speeded up??
			out[id] = (float) t;
		}
	}
	// A = A/B;
	__global__ void ElementwiseNorm(float * A , float  *B, int size) {
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<size)
			A[id] /= B[id];
	}
	//
	// get idx of max value... this is written in a stupid way using num of blocks & threads=1
	//
	__global__ void getIdxOfMaxVal_naive(const float * A , float * out , int size) {
		float max = -FLT_MAX;
		float tmp = -1;
		for (int i=0 ; i<size ; i++){
			if (A[i]>max){
				max   = A[i];
				tmp = i;
			}
		}
		*out = tmp;
	}
	__global__ void getIdxOfMinVal_naive(const float * A , float * out , int size) {
		float min = FLT_MAX;
		float tmp = -1;
		for (int i=0 ; i<size ; i++){
			if (A[i]<min){
				min   = A[i];
				tmp = i;
			}
		}
		*out = tmp;
	}

	// get row of C matrix
	// efficiency should imnprove when using __sync!  but it shouldne matter only for big matrices.. :)
	__global__ void getRow_naive(const float * A , float * row_id , float * out , int Acols) {
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<Acols){
			out[id] = A[id+(int)(*row_id)*Acols];
		}
	}
	__global__ void getRow_IntId_naive(const float * A , int row_id , float * out , int Acols) {
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<Acols){
			out[id] = A[id+row_id*Acols];
		}
	}
	__global__ void Matrix_getCol_FloatId_naive(const float * A , int Acount, int Acols, float * out0 , int out0count, int out0cols, float col_id) {
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<out0count){
			out0[id] = A[id*Acols + (int)col_id];
		}
	}
	__global__ void Matrix_getRow_FloatId_naive(const float * A , int Acount, int Acols, float * out0 , int out0count, int out0cols, float row_id) {
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<Acols){
			out0[id] = A[id+(int)row_id*Acols];
		}
	}



	__global__ void NN_naive(float * A , int colsA , int sizeA , float * B , int colsB , int numsB , int dim , float * idx , float * dist){
		float tmp_dist = 99999;
		int nn_id = -1;
		int idA = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;

		for (int idB = 0 ; idB<(numsB*colsB) ; idB += colsB){
			float adist = Dist_between_two_vec(A+colsA*idA , B+colsB*idB , dim);
			if (tmp_dist>adist){
				tmp_dist = adist;
				nn_id = idB;
			}
		}
		*(dist+idA) = tmp_dist;
		*(idx+idA)  = nn_id;
	}















}
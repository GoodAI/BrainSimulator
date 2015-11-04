/*
*
*   To work with matrices.
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

#include <iostream>




extern "C" 
{



	__global__  void MatrixCopy_naive (const float * A , int Acount, int Acols, float * out0 , int out0count)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<out0count)
		{
			out0[id] = A[id];
		}
	}
	__global__ void Matrix_getRow_FloatId_naive(const float * A , int Acount, int Acols, float * out0 , int out0count, int out0cols, const float row_id)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<Acols)
		{
			out0[id] = A[id+(int)row_id*Acols];
		}
	}
	__global__ void Matrix_getRow_FloatPointer_naive(const float * A , int Acount, int Acols, const float * rowId , int empty_par1, int empty_par2, float * out0 , int out0count, int out0cols)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<Acols)
		{
			out0[id] = A[id + (int)rowId[0]*Acols];
		}
	}
	__global__ void Matrix_getCol_FloatId_naive(const float * A , int Acount, int Acols, float * out0 , int out0count, int out0cols, float col_id)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<Acount/Acols)
		{
			out0[id] = A[id*Acols + (int)col_id];
		}
	}
	__global__ void Matrix_getCol_FloatPointer_naive(const float * A , int Acount, int Acols, const float * colId , int empty_par1, int empty_par2, float * out0 , int out0count, int out0cols)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<Acount/Acols)
		{
			out0[id] = A[id*Acols + (int)colId[0]];
		}
	}
	
	__global__ void Matrix_cos_naive(const float * A , int Acount, int Acols, float * out0 , int out0count)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<out0count){
			out0[id] = cos(A[id]);
		}
	}
	__global__ void Matrix_sin_naive(const float * A , int Acount, int Acols, float * out0 , int out0count)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<out0count){
			out0[id] = sin(A[id]);
		}
	}



    __global__ void Matrix_transposeFromSVDnodeCOPY(const float* A, int Acount, int Acols, float* out0)
	{
        int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;

        int Arows = Acount/Acols;

		int x = id / Arows;
		int y = id % Arows;

		if (id < Acount)
		{
			out0[x * Arows + y] = A[y * Acols + x];
		}
	}


    /*
     * this works:
     * mat.*mat ...as expected
     * mat.*vec ...vec (B) can be row or column then each row or column of matrix (A) will be columnt wise multiplied with B.
     */

    __global__ void Matrix_MultiplElementWise_naiveOLD(const float * A , int Acount, int Acols, const float * B , int Bcount, int Bcols, float * out0 , int out0count, int out0cols)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
        int id_row,id_col;
		if (id<Acount)
		{
            if (Acount==Bcount) // matrix .* matrix
            {
                out0[id] = A[id]*B[id];
            }
            else if (Bcols==1) // matrix .* row vector
            {
                id_row = id/Acols;
                out0[id] = A[id]*B[id_row];
            }
            else // matrix .* column vector
            {
                id_col = id%Acols;
                out0[id] = A[id]*B[id_col];
            }
		}
	}



    __device__ float Perform_operation (char op, float A, float B){
        if (op=='+')
            return A+B;
        if (op=='-')
            return A-B;
        if (op=='*')
            return A*B;
		if (op == '^')
			return powf(A,B);
        return -1;
    }
    __device__ void Matrix_performOperation_naive(char op, int id,  const float * A , int Acount, int Acols, const float * B , int Bcount, int Bcols, float * out0 , int out0count, int out0cols, float value)
	{
        if (Bcount == 0){
            out0[id] = Perform_operation(op,A[id],value);
        }
        else if (Bcount == 1){
            out0[id] = Perform_operation(op,A[id],B[0]);
        }
        else if (Acount==Bcount) // matrix .* matrix
        {
            out0[id] = Perform_operation(op,A[id],B[id]);
        }
        else if (Bcols == 1) // matrix .* row vector
        {
            int id_row = id/Acols;
            out0[id] = Perform_operation(op,A[id],B[id_row]);
        }
        else // matrix .* column vector
        {
           int id_col = id%Acols;
           out0[id] = Perform_operation(op,A[id],B[id_col]);
        }
	}
__global__ void Matrix_Addition_naive(const float * A , int Acount, int Acols, const float * B , int Bcount, int Bcols, float * out0 , int out0count, int out0cols, float value)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<Acount)
		{
            Matrix_performOperation_naive('+',id,A ,Acount, Acols, B , Bcount, Bcols, out0 , out0count, out0cols, value);
		}
	}
__global__ void Matrix_Pow_naive(const float * A, int Acount, int Acols, const float * B, int Bcount, int Bcols, float * out0, int out0count, int out0cols, float value)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x + blockDim.x*blockIdx.x + threadIdx.x;
		if (id<Acount)
		{
			Matrix_performOperation_naive('^', id, A, Acount, Acols, B, Bcount, Bcols, out0, out0count, out0cols, value);
		}
	}
__global__ void Matrix_MultiplElementWise_naive(const float * A , int Acount, int Acols, const float * B , int Bcount, int Bcols, float * out0 , int out0count, int out0cols, float value)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<Acount)
		{
            Matrix_performOperation_naive('*',id,A ,Acount, Acols, B , Bcount, Bcols, out0 , out0count, out0cols, value);
		}
	}
__global__ void Matrix_Substraction_naive(const float * A , int Acount, int Acols, const float * B , int Bcount, int Bcols, float * out0 , int out0count, int out0cols, float value)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<Acount)
		{
            Matrix_performOperation_naive('-',id,A ,Acount, Acols, B , Bcount, Bcols, out0 , out0count, out0cols,value);
		}
	}





	__global__ void AbsKernel_naive(const float * A , int Acount, int Acols, float * out0 , int out0count)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x	+ blockDim.x * blockIdx.x	+ threadIdx.x;
		if(id < out0count)
		{	
			out0[id] = fabsf(A[id]);
		}
	}		
	__global__ void RoundKernel_naive(const float * A , int Acount, int Acols, float * out0 , int out0count)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x	+ blockDim.x * blockIdx.x	+ threadIdx.x;
		if(id < out0count)
		{
			out0[id] = round(A[id]);
		}
	}
	__global__ void ExpKernel_naive(const float * A , int Acount, int Acols, float * out0 , int out0count)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x	+ blockDim.x * blockIdx.x	+ threadIdx.x;
		if(id < out0count)
		{
			out0[id] = exp(A[id]);
		}
	}
    __global__ void LogKernel_naive(const float * A , int Acount, int Acols, float * out0 , int out0count)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x	+ blockDim.x * blockIdx.x	+ threadIdx.x;
		if(id < out0count)
		{
			out0[id] = log(A[id]);
		}
	}
	__global__ void FloorKernel_naive(const float * A , int Acount, int Acols, float * out0 , int out0count)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x	+ blockDim.x * blockIdx.x	+ threadIdx.x;
		if(id < out0count)
		{
			out0[id] = floor(A[id]);
		}
	}	
	__global__ void CeilKernel_naive(const float * A , int Acount, int Acols, float * out0 , int out0count)
	{		
		int id = blockDim.x * blockIdx.y * gridDim.x	+ blockDim.x * blockIdx.x	+ threadIdx.x;
		if(id < out0count)
		{
			out0[id] = ceil(A[id]);
		}
	}	





    __global__ void Matrix_PermuteRows(const float * A , int Acount, int Acols, const float * B , int Bcount, int Bcols, float * out0 , int out0count, int out0cols)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+   blockDim.x*blockIdx.x	  +   threadIdx.x;
        int id_row, id_col, id_rowNew;
		if (id<Acount)
		{
            id_row = id/Acols;
            id_col = id%Acols;
            id_rowNew = B[id_row]*Acols;
            out0[id] = A[id_col + id_rowNew];
		}
	}


}



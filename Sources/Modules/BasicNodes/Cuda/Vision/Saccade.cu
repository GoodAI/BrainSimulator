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


#include "VisionMath.cu"





extern "C" 
{

	



	

	__global__ void UdpateEnergyTerm_movement(
		float* energy, 
		int energy_dim, 
		int nPatches, 
		float * desc,
		int desc_dim,
		int id_desc_move) // whic hindex is the one with movement
	{

		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		int idDim   = id % energy_dim;
		int idPatch = id / energy_dim;
		if (id<energy_dim*nPatches){
			if (idDim==1) // movement
				energy[id] = -desc[idPatch*desc_dim + id_desc_move];
		}
	}



	__global__ void UdpateEnergyTerm_time(
		float* energy, 
		int energy_dim, 
		int nPatches, 
		float * idFocuser_focused , 
		float par_time_increase_energy_on_focus,
		float par_time_decrease_energy_in_time)
	{

		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		int idDim   = id % energy_dim;
		int idPatch = id / energy_dim;
		if (id<energy_dim*nPatches){
			if (idDim==0){ // time
				if (idPatch==(int)(*idFocuser_focused)) // it is id that focuser just focused
					energy[id] += par_time_increase_energy_on_focus;
				else
					energy[id] /= par_time_decrease_energy_in_time ;
			}
		}
	}




	// sum energy terms and save them into toal (each block calculates each element , each thread goes for the term)
	// it is written in the way to be optimized for large images (with shared...)
	__global__ void ComputeTotalEnergy(
		float* EnTerms,
		float * EnTotal,
		int nPatches,
		int nTerms,
		float par_alpha_how_to_prefer_movememnt)
	{

		int id_patch = blockIdx.x;
		int id_dim = threadIdx.x;
		__shared__ float actual_en;
		float alpha; //. how to weight differnet terms
		if (id_dim==1)
			alpha = par_alpha_how_to_prefer_movememnt;
		else if (id_dim==0)
			alpha = 1-par_alpha_how_to_prefer_movememnt;
		else
			alpha = 1.0f;
		actual_en = 0; //. actual energy

		__syncthreads;

		atomicAdd(&actual_en , alpha*EnTerms[id_patch*nTerms + id_dim]);

		__syncthreads;

		if (id_dim==0)
			EnTotal[id_patch] = actual_en;	
	}


		
	__global__ void ReeeigthTotalTermsThatBrainDoesNotFocusOn(
		float * EnTotal,
		int nPatches,
		float * XYS,
		int  XYS_dim,
		float * brains_focus)
	{

		int idTerm = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	   + threadIdx.x;
		if (idTerm<nPatches){
			float dist = Dist_between_two_vec(XYS+idTerm*XYS_dim , brains_focus, 2);
			if (dist>brains_focus[2]){ //. check wheter the center of the object is further than the brain's limit
				EnTotal[idTerm] = FLT_MAX;	//. set its energy to somehting high -> thus not interesting :)
			}
			

		}
	}
	





}





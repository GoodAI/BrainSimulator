#define _SIZE_T_DEFINED 

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>


#include "../Observers/ColorHelpers.cu"



extern "C" 
{

	


	__global__  void FillImByEnergy (unsigned int * texture , float* mask , int size_im , float* energiesPerElement , float max_energy){
		int idx = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (idx<size_im){
			int id_element = mask[idx];
			float energy   = energiesPerElement[id_element];
			texture[idx]   = hsva_to_uint_rgba(1.0f, .0f, energy/max_energy, 1.0f);

		}
	}




	//--- 2 remamber where focuser is shooting...
	__global__  void FocuserTracker (unsigned int * texture , int size_im , int dim_im,  float* tracker_storage , int tracker_storageSize , float * id_focuser , float * patches , int patch_dim){
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		int tr_elements = (tracker_storageSize-1)/2;
		int tr_dim = 2;
		if (id<size_im){
			float x  = patches[0+patch_dim*((int)*id_focuser)];
			float y  = patches[1+patch_dim*((int)*id_focuser)];

			int trackerID_last = (int)tracker_storage[0]; // it is circular list :)
			int trackerID_this = trackerID_last+1;
			if (trackerID_this>tr_elements)
				trackerID_this = 0;
			tracker_storage[1+trackerID_this*tr_dim+0] = x; // first element is where last one was saved!
			tracker_storage[1+trackerID_this*tr_dim+1] = y;
			tracker_storage[0] = trackerID_this;
			
			texture[id] = GET_RGBA(50 , 10 , 100 , 100);

			for (int i=0 ; i<trackerID_last ; i++){
				x = tracker_storage[1+i*tr_dim+0];
				y = tracker_storage[1+i*tr_dim+1];
				texture[(int)x+(int)y*dim_im] = GET_RGBA(255, 255 , 255 , 100);
				texture[(int)x+1+(int)y*dim_im] = GET_RGBA(255, 255 , 255 , 100);
			}
		}
	}



}
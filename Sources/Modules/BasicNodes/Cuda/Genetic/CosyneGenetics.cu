//Includes for IntelliSense 
#define _SIZE_T_DEFINED

#include <cuda.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include "float.h"
#include <math.h>
#include <stdarg.h>
#include <stdio.h>

#define PI acos(-1.0)


extern "C"{

	// Write coefficients back into the matrix, ready for fitness evaluation/ Inverse DCT
	__global__ void implantCoeffs(float* matrices, float *coeffArray, int savedCoeffs, int dimsize){

		int id = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
			+ threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

		int offsetMatrix = id * dimsize * dimsize,
			offsetCoeff = id * savedCoeffs,
			coeffsLeft = savedCoeffs,
			x, y, y_n = 0, x_n = 1,
			numberinrow, tmp;

		matrices[offsetMatrix] = coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)];
		coeffsLeft -= 1;

		while (coeffsLeft > 0){
			// Work out number in row
			x = x_n;
			y = y_n;

			if (x_n < dimsize - 1){
				numberinrow = x_n + 1;
			}
			else{
				numberinrow = x_n - (y_n - 1);
			}

			if (numberinrow % 2 == 0){
				// Even
				while (numberinrow > 0 && coeffsLeft > 0){
					matrices[offsetMatrix + x + y * dimsize] = coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)];
					numberinrow--;
					coeffsLeft--;

					if ((numberinrow + 1) % 2 == 0){
						// Swap x and y
						tmp = x;
						x = y;
						y = tmp;
					}
					else{
						// Swap x and y
						tmp = x;
						x = y;
						y = tmp;
						x--;
						y++;
					}
				}
			}
			else{
				// Odd
				while (numberinrow > 1 && coeffsLeft > 0){
					matrices[offsetMatrix + x + y * dimsize] = coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)];
					numberinrow--;
					coeffsLeft--;
					if ((numberinrow + 1) % 2 == 1){
						// Swap x and y
						tmp = x;
						x = y;
						y = tmp;
					}
					else{
						// Swap x and y
						tmp = x;
						x = y;
						y = tmp;
						x--;
						y++;
					}
				}
				if (coeffsLeft > 0){
					// add the odd one
					matrices[offsetMatrix + x + y * dimsize] = coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)];
					numberinrow--;
					coeffsLeft--;
				}
			}
			if (x_n == dimsize - 1){
				y_n++;
			}
			else{
				x_n++;
			}
		}

	}


	// Creates a square cosine matrix and its inverse
	__global__ void createCosineMatrix(float* matrix, int xsize){
		int threadGlobalID = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
			+ threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

		int i;
		for (i = 0; i < xsize; i++){
			if (threadGlobalID == 0)
				matrix[threadGlobalID + i * xsize] = 1 / sqrt((float)xsize);
			else
				matrix[threadGlobalID + i * xsize] = (sqrt((float)2 / xsize) * cos((PI * (2 * i + 1) * threadGlobalID) / (2 * xsize)));
		}
	}

	// This is obscenely complex for something so seemingly simple
	// Each thread, extracts savedCoeffs from a matrix, assumes square martix
	__global__ void extractCoeffs(const float  *matrices, float *coeffArray, int savedCoeffs, int dimsize){
		int threadGlobalID = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
			+ threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

		int offsetMatrix = threadGlobalID * dimsize * dimsize,
			offsetCoeff = threadGlobalID * savedCoeffs,
			coeffsLeft = savedCoeffs,
			x, y, y_n = 0, x_n = 1,
			numberinrow, tmp;

		coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)] = matrices[offsetMatrix];
		coeffsLeft -= 1;

		while (coeffsLeft > 0){
			// Work out number in row
			x = x_n;
			y = y_n;

			if (x_n < dimsize - 1)
				numberinrow = x_n + 1;
			else
				numberinrow = x_n - (y_n - 1);

			if (numberinrow % 2 == 0){
				// Even
				while (numberinrow > 0 && coeffsLeft > 0){
					coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)] = matrices[offsetMatrix + x + y * dimsize];
					numberinrow--;
					coeffsLeft--;

					if ((numberinrow + 1) % 2 == 0){
						// Swap x and y
						tmp = x;
						x = y;
						y = tmp;
					}
					else{
						// Swap x and y
						tmp = x;
						x = y;
						y = tmp;
						x--;
						y++;
					}
				}
			}
			else{
				// Odd
				while (numberinrow > 1 && coeffsLeft > 0){
					coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)] = matrices[offsetMatrix + x + y * dimsize];
					numberinrow--;
					coeffsLeft--;
					if ((numberinrow + 1) % 2 == 1){
						// Swap x and y
						tmp = x;
						x = y;
						y = tmp;
					}
					else{
						// Swap x and y
						tmp = x;
						x = y;
						y = tmp;
						x--;
						y++;
					}
				}
				if (coeffsLeft > 0){
					// add the odd one
					coeffArray[offsetCoeff + (savedCoeffs - coeffsLeft)] = matrices[offsetMatrix + x + y * dimsize];
					numberinrow--;
					coeffsLeft--;
				}
			}
			if (x_n == dimsize - 1){
				y_n++;
			}
			else{
				x_n++;
			}
		}
	}

	// Generates chromSize random numbers between alpha and -alpha and stores them in the chromosomes array
	__global__ void generateCoefficients(float *chromosomes, const int chromSize, const float* noise, const int population, const int alpha){

		int i;

		// For up to a 1D grid of 3D blocks...
		int threadGlobalID = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
			+ threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

		curandState st;
		curand_init((int)noise[threadGlobalID] << threadGlobalID, threadGlobalID * (threadGlobalID == population - 1 ? noise[0] : noise[threadGlobalID]), 0, &st);

		if (threadGlobalID > 0){
			for (i = 0; i < chromSize; i++){
				if (curand_uniform(&st) < 0.5){
					chromosomes[chromSize*threadGlobalID + i] = curand_uniform(&st) *alpha;
				}
				else{
					chromosomes[chromSize*threadGlobalID + i] = -1 * curand_uniform(&st) * alpha;
				}
			}
		}
	}

	// Performs the CoSyNE genetic algorithm.
	// -- Replace all non-survivors with crossover from two random parents
	// -- Randomly mutate the new population members 
	// -- Permute the genes of the chromosome population
	__global__ void grow(float *matrices, const int dimension, const int coefficients, const int population, float *chromosomes, const float * noise, const float mutationRate, const int kept, const float* fitnesses, int *mark, const int alpha){

		int i, wloc;

		curandState st;

		// For up to a 1D grid of 3D blocks...
		int threadGlobalID = blockIdx.x * blockDim.x * blockDim.y * blockDim.z
			+ threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;

		int chromOffset = threadGlobalID * coefficients;
		int parent1, parent2, point;
		float tmp1, tmp2;

		// Init the random number generator
		curand_init((int)noise[threadGlobalID] << threadGlobalID, threadGlobalID * (threadGlobalID == population - 1 ? noise[0] : noise[threadGlobalID]), 0, &st);

		// Repopulate
		// The threads with the keepmask are kept, all others are replaced with crossovers
		if (threadGlobalID > kept - 1){
			// pick two parents -- 0 is not included in the random distribution
			parent1 = floor(curand_uniform(&st) * kept);
			parent2 = floor(curand_uniform(&st) * kept);
			//pick a point on the chromosome
			point = floor(curand_uniform(&st) * coefficients);
			for (i = 0; i < point; i++){
				chromosomes[chromOffset + i] = chromosomes[parent1 * coefficients + i];
			}
			//Copy past the point for parent 2
			for (i = point; i < coefficients; i++){
				chromosomes[chromOffset + i] = chromosomes[parent2 * coefficients + i];
			}
		}

		// Mutate children
		if (threadGlobalID > kept - 1){
			for (i = 0; i < coefficients; i++){
				if (curand_uniform(&st) <= mutationRate){
					if (curand_uniform(&st) < 0.5){
						chromosomes[chromOffset + i] = curand_uniform(&st) * -1 * alpha;
					}
					else{
						chromosomes[chromOffset + i] = curand_uniform(&st) * alpha;
					}
				}
			}
		}

		// Permute
		if (threadGlobalID < coefficients){
			// Mark genes for permutation
			for (i = 0; i < population; i++){
				if (curand_uniform(&st) < (1 - sqrt((fitnesses[i] - fitnesses[population - 1]) / (fitnesses[0] - fitnesses[population - 1])))){
					mark[coefficients * i + threadGlobalID] = 1;
				}
				else{
					mark[coefficients * i + threadGlobalID] = 0;
				}
			}

			wloc = -1;
			// Permute selected genes
			for (i = 0; i < population; i++){
				if (mark[coefficients * i + threadGlobalID] == 1){
					if (wloc == -1){
						wloc = i;
						tmp1 = chromosomes[coefficients * i + threadGlobalID];
					}
					else{
						tmp2 = chromosomes[coefficients * i + threadGlobalID];
						chromosomes[coefficients * i + threadGlobalID] = tmp1;
						tmp1 = tmp2;
					}
				}
			}
			if (wloc != -1){
				chromosomes[coefficients * wloc + threadGlobalID] = tmp1;
			}
		}

		__syncthreads();
		//Place into relevant matrix
		for (i = 0; i < dimension*dimension; i++){
			matrices[threadGlobalID * dimension * dimension + i] = 0.0f;
		}
	}

}
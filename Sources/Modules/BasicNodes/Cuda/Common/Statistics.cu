#define _SIZE_T_DEFINED 

#include <cuda.h> 
#include <math_constants.h>
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>

#include "../Common/helper_math.h"

extern "C"  
{	
	__global__ void PrepareMeanStdDev(float* input, float* delta, int imageWidth, int imageHeight)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x				
			+ threadIdx.x;

		int size = imageWidth * imageHeight;
			
		if (id < size)
		{
			int px = id % imageWidth;
			int py = id / imageWidth;

			float2 pixPos = {  2.0f * px / imageWidth - 1,  2.0f * py / imageHeight - 1};

			//mean sum
			delta[id] = input[id] * pixPos.x;
			delta[id + size] = input[id] * pixPos.y;	

			//variance sum
			delta[id + 2 * size] = input[id] * pixPos.x * pixPos.x;	
			delta[id + 3 * size] = input[id] * pixPos.y * pixPos.y;	
		}
	}

	__global__ void FinalizeMeanStdDev(float* stats) 
	{
		float sumWeights = stats[4];

		if (sumWeights > 0) 
		{
			float2 mean = { stats[0] / sumWeights, stats[1] / sumWeights };

			float2 stdDev = { 
				(stats[2] * sumWeights - stats[0] * stats[0]) / (sumWeights * sumWeights), 
				(stats[3] * sumWeights - stats[1] * stats[1]) / (sumWeights * sumWeights),  
			};

			stats[0] = mean.x;
			stats[1] = mean.y;

			stats[2] = stdDev.x;
			stats[3] = stdDev.y;			
		}
		else 
		{
			stats[0] = 0;
			stats[1] = 0;

			stats[2] = 0;
			stats[3] = 0;	
		}
	}
	
	const int NUM_SUMS = 5;
	const int MAX_CENTROIDS = 256;

	typedef struct {
		float X;
		float Y;
		float VarianceX;
		float VarianceY;
		float Weight;
		float DBI; //Davies-Bouldin Index
	} Centroid;

	__global__ void PrepareK_Means(float* input, Centroid* centroids, int numOfCentroids, float* delta, int imageWidth, int imageHeight)
	{
		 __shared__ float smem_centroids[MAX_CENTROIDS * 2];

		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x				
			+ threadIdx.x;

		int size = imageWidth * imageHeight;

		
		if (threadIdx.x < numOfCentroids) 
		{
			smem_centroids[2 * threadIdx.x] = centroids[threadIdx.x].X;
			smem_centroids[2 * threadIdx.x + 1] = centroids[threadIdx.x].Y;
		}
		__syncthreads();
		

		if (id < size)
		{
			int px = id % imageWidth;
			int py = id / imageWidth;			
			float pixelValue = input[id];

			if (pixelValue > 0) 
			{
				float2 pixPos = {  2.0f * px / imageWidth - 1,  2.0f * py / imageHeight - 1};

				float2 centroid = { smem_centroids[0], smem_centroids[1]};				
				centroid -= pixPos;

				float minDistSq = dot(centroid, centroid); 	
				int minCentroidIdx = 0;

				for (int i = 1; i < numOfCentroids; i++) 
				{
					centroid.x = smem_centroids[2 * i];
					centroid.y = smem_centroids[2 * i + 1];					
					centroid -= pixPos;

					float distSq = dot(centroid, centroid);

					if (distSq < minDistSq) 
					{
						minDistSq = distSq;
						minCentroidIdx = i;
					}
				}

				//minCentroid deltas

				//w * pos
				delta[minCentroidIdx * NUM_SUMS * size + 0 * size + id] = pixelValue * pixPos.x;
				delta[minCentroidIdx * NUM_SUMS * size + 1 * size + id] = pixelValue * pixPos.y;	

				//w * pos^2
				delta[minCentroidIdx * NUM_SUMS * size + 2 * size + id] = pixelValue * pixPos.x * pixPos.x;
				delta[minCentroidIdx * NUM_SUMS * size + 3 * size + id] = pixelValue * pixPos.y * pixPos.y;	

				//w
				delta[minCentroidIdx * NUM_SUMS * size + 4 * size + id] = pixelValue;	
			}
		}
	}

	__global__ void SumCentroids(float* delta, float* sumDelta, int numOfCentroids, int numOfElements)	
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x				
			+ threadIdx.x;

		if (id < numOfCentroids * NUM_SUMS) 
		{
			float sum = 0;

			for (int i = 0; i < numOfElements; i++) 
			{
				sum += delta[numOfElements * id + i];
			}

			sumDelta[id] = sum;
		}
	}

	__global__ void FinalizeK_Means(Centroid* centroids, int numOfCentroids, float* sumDelta, float learningFactor, int offset) 
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x				
			+ threadIdx.x;

		if (id < numOfCentroids) 
		{
			float sumWeights = sumDelta[NUM_SUMS * (id + offset) + 4];
			float2 sumMean = { sumDelta[NUM_SUMS * (id + offset)], sumDelta[NUM_SUMS * (id + offset) + 1] }; 

			centroids[id + offset].Weight = sumWeights;

			if (sumWeights > 0) {
				
				centroids[id + offset].X = sumMean.x / sumWeights;
				centroids[id + offset].Y = sumMean.y / sumWeights;
				centroids[id + offset].VarianceX = sqrtf((sumDelta[NUM_SUMS * (id + offset) + 2] * sumWeights - sumMean.x * sumMean.x) / (sumWeights * sumWeights));
				centroids[id + offset].VarianceY = sqrtf((sumDelta[NUM_SUMS * (id + offset) + 3] * sumWeights - sumMean.y * sumMean.y) / (sumWeights * sumWeights));				
				centroids[id + offset].DBI = centroids[id + offset].VarianceX * centroids[id + offset].VarianceY;
			}

			else {				
				centroids[id + offset].X = CUDART_INF_F;
				centroids[id + offset].Y = CUDART_INF_F;				
				centroids[id + offset].VarianceX = 0;
				centroids[id + offset].VarianceY = 0; 
				centroids[id + offset].DBI = CUDART_INF_F;
			}
		}
	}

	__global__ void Prepare_2_MeansForDivision(float* input, Centroid* centroids, int c_n1, int c_n2, int c_src, float* delta, int imageWidth, int imageHeight)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x				
			+ threadIdx.x;

		int size = imageWidth * imageHeight;
			
		if (id < size)
		{
			int px = id % imageWidth;
			int py = id / imageWidth;

			bool insideSrc = delta[c_src * NUM_SUMS * size + 4 * size + id] != 0;

			if (input[id] > 0 && insideSrc) {

				float2 pixPos = {  2.0f * px / imageWidth - 1,  2.0f * py / imageHeight - 1};

				float2 centroid1 = { centroids[c_n1].X, centroids[c_n1].Y };
				centroid1 -= pixPos;

				float2 centroid2 = { centroids[c_n2].X, centroids[c_n2].Y };
				centroid2 -= pixPos;

				float distSqC1 = dot(centroid1, centroid1);				
				float distSqC2 = dot(centroid2, centroid2); 
				
				int minCentroidIdx = distSqC1 > distSqC2 ? c_n2 : c_n1;
				
				//w * pos
				delta[minCentroidIdx * NUM_SUMS * size + 0 * size + id] = input[id] * pixPos.x;
				delta[minCentroidIdx * NUM_SUMS * size + 1 * size + id] = input[id] * pixPos.y;	

				//w * pos^2
				delta[minCentroidIdx * NUM_SUMS * size + 2 * size + id] = input[id] * pixPos.x * pixPos.x;
				delta[minCentroidIdx * NUM_SUMS * size + 3 * size + id] = input[id] * pixPos.y * pixPos.y;	

				//w
				delta[minCentroidIdx * NUM_SUMS * size + 4 * size + id] = input[id];	
			}
		}
	}

	__global__ void Prepare_1_MeansForJoin(float* input, int c_src1, int c_src2, int c_n, float* delta, int imageWidth, int imageHeight)
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x				
			+ threadIdx.x;

		int size = imageWidth * imageHeight;
			
		if (id < size)
		{
			int px = id % imageWidth;
			int py = id / imageWidth;

			bool insideSrc1 = delta[c_src1 * NUM_SUMS * size + 4 * size + id] != 0;
			bool insideSrc2 = delta[c_src2 * NUM_SUMS * size + 4 * size + id] != 0;

			if (input[id] > 0 && (insideSrc1 || insideSrc2)) {

				float2 pixPos = {  2.0f * px / imageWidth - 1,  2.0f * py / imageHeight - 1};
				
				//w * pos
				delta[c_n * NUM_SUMS * size + 0 * size + id] = input[id] * pixPos.x;
				delta[c_n * NUM_SUMS * size + 1 * size + id] = input[id] * pixPos.y;	

				//w * pos^2
				delta[c_n * NUM_SUMS * size + 2 * size + id] = input[id] * pixPos.x * pixPos.x;
				delta[c_n * NUM_SUMS * size + 3 * size + id] = input[id] * pixPos.y * pixPos.y;	

				//w
				delta[c_n * NUM_SUMS * size + 4 * size + id] = input[id];	
			}	
			else 
			{
				delta[c_n * NUM_SUMS * size + 0 * size + id] = 0;
				delta[c_n * NUM_SUMS * size + 1 * size + id] = 0;	

				//w * pos^2
				delta[c_n * NUM_SUMS * size + 2 * size + id] = 0;
				delta[c_n * NUM_SUMS * size + 3 * size + id] = 0;	

				//w
				delta[c_n * NUM_SUMS * size + 4 * size + id] = 0;
			}
		}
	}	

	__global__ void EvaluateDBI(Centroid* centroids, float* CxC_DBI, int numOfCentroids) 
	{
		int id = blockDim.x * blockIdx.y * gridDim.x
			+ blockDim.x * blockIdx.x				
			+ threadIdx.x;

		int c_i = id % numOfCentroids;
		int c_j = id / numOfCentroids;
		int size = numOfCentroids * numOfCentroids;

		if (id < size) 
		{
			if (c_i != c_j) 
			{
				float2 distV = { centroids[c_i].X - centroids[c_j].X, centroids[c_i].Y - centroids[c_j].Y };				
				CxC_DBI[c_j * MAX_CENTROIDS + c_i] = 0.5 * (centroids[c_i].DBI + centroids[c_j].DBI) / length(distV);
			}
			else 
			{
				CxC_DBI[c_j *  MAX_CENTROIDS + c_i] = CUDART_INF_F;
			}
		}
	}
}
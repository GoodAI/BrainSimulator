#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

//Includes for IntelliSense 
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif

extern "C"  
{
	__global__ void InitCentroidsKernel(
		
		float *centroidCoordinates,
		float *randomNumbers,
		float minX,
		float maxX,
		float minY,
		float maxY,
		int centroids

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < centroids)
		{
			centroidCoordinates[threadId *2] = (maxX - minX) * randomNumbers[threadId * 2] + minX;
			centroidCoordinates[threadId * 2 + 1] = (maxY - minY) * randomNumbers[threadId * 2 + 1] + minY;
		}
	}

	__global__ void ComputeEuklidianDistancesKernel(
		
		float *inputImg,
		int imgWidth,
		int imgHeight,
		float *centroidCoordinates,
		float *distanceMatrix,
		int centroids,
		int inputSize
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < inputSize)
		{
			int pointX = threadId % imgWidth;
			int pointY = threadId / imgWidth;

			float X = (float)pointX;
			float Y = (float)pointY;

			float dist;

			float centroidX;
			float centroidY;

			for(int c = 0; c < centroids; c++)
			{
				centroidX = centroidCoordinates[c * 2];
				centroidY = centroidCoordinates[c * 2 + 1];

				dist = sqrtf( (centroidX - X) * (centroidX - X) + (centroidY - Y) * (centroidY - Y) );
				distanceMatrix[c * inputSize + threadId] = dist;
			}
		}
	}

	__global__ void FindNearestCentroidKernel(
		
		float *distanceMatrix,
		int *nearestCentroid,
		int centroids,
		int inputSize
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < inputSize)
		{
			float min = FLT_MAX;
			float dist;
			for(int c = 0; c < centroids; c++)
			{
				dist = distanceMatrix[c * inputSize + threadId];
				if(dist <= min)
				{
					nearestCentroid[threadId] = c;
					min = dist;
				}
			}
		}
	}

	__global__ void SumNewCentroidCoordinatesKernel(
		float *input,
		int imgWidth,
		int imgHeight,
		float *centroidCoordinates,
		int *nearestCentroid,
		float *pointsWeight,
		int inputSize
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		if(threadId < inputSize)
		{
			int pointX = threadId % imgWidth;
			int pointY = threadId / imgWidth;

			float X = (float)pointX;
			float Y = (float)pointY;

			int centroidId = nearestCentroid[threadId];

			float weight = input[threadId];

			atomicAdd(&centroidCoordinates[centroidId * 2], weight * X);
			atomicAdd(&centroidCoordinates[centroidId * 2 + 1], weight * Y);
			atomicAdd(&pointsWeight[centroidId], weight);
		}
	}
	

	__global__ void AvgCentroidCoordinatesKernel(
		
		float *centroidCoordinates,
		float *pointsWeight,
		int inputSize,
		int centroids
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < centroids * 2)
		{
			if(pointsWeight[threadId / 2] == 0.00f)
			{
				centroidCoordinates[threadId] = 0.00f;
			}
			else
			{
				centroidCoordinates[threadId] = centroidCoordinates[threadId] / pointsWeight[threadId / 2];
			}
		}
	}

	__global__ void CopyInputToVisFieldKernel(
	
		float *input,
		float *visField,

		int inputSize
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < inputSize)
		{
			visField[threadId] = input[threadId];
		}
	}

	__global__ void MarkCentroidsKernel(
		float *centroidCoordinates,
		float *visField,
		int imgWidth,
		int imgHeight,
		int centroids
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
						+ blockDim.x*blockIdx.x				//blocks preceeding current block
						+ threadIdx.x;
		if(threadId < centroids)
		{
			int x = lrintf(centroidCoordinates[threadId * 2]);
			int y = lrintf(centroidCoordinates[threadId * 2 + 1]);
			
			visField[y * imgWidth + x] = -1.00f;
		
		}
	}
}
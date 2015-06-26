#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

#include "../Observers/ColorScaleObserverSingle.cu"

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

	// PHYSICS PART ----------------------------

	__global__ void SetForcesToZeroKernel(
		
		float *force,
		int maxCells

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < maxCells * 3)
		{
			force[threadId] = 0.00f;	
		}
	}



	__global__ void SpringKernel(
		
		int *activityFlag,
		int *connectionMatrix,
		float *pointsCoordinates,
		float springStrength,
		float *force,
		int maxCells
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < maxCells)
		{
			if(activityFlag[threadId] == 1)
			{
				int n;
				float x,y,z;
				float nX, nY, nZ;
				float dX, dY, dZ;
				float fX, fY, fZ;
				float distanceSquared, distance;

				x = pointsCoordinates[threadId * 3];
				y = pointsCoordinates[threadId * 3 + 1];
				z = pointsCoordinates[threadId * 3 + 2];

				for(n = 0; n < maxCells; n++)
				{
					if(connectionMatrix[threadId * maxCells + n] == 1)
					{
						nX = pointsCoordinates[n * 3];
						nY = pointsCoordinates[n * 3 + 1];
						nZ = pointsCoordinates[n * 3 + 2];

						dX = nX - x;
						dY = nY - y;
						dZ = nZ - z;

						if(dX != 0 || dY != 0 || dZ != 0)
						{
							distanceSquared = dX * dX + dY * dY + dZ * dZ;
							distance = sqrtf(distanceSquared);

							fX = springStrength * dX;
							fY = springStrength * dY;
							fZ = springStrength * dZ;

							force[threadId * 3] += fX;
							force[threadId * 3 + 1] += fY;
							force[threadId * 3 + 2] += fZ;
						}
					}
				}
			
			}

		}
	}

	__global__ void RepulsionKernel(
		
		float repulsion,
		float repulsionDistance,
		float *force,
		float *pointsCoordinates,
		int *activityFlag,
		int maxCells
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < maxCells)
		{
			if(activityFlag[threadId] == 1)
			{
				float x,y,z;
				float nX, nY, nZ;
				float dX, dY, dZ;
				float fX, fY, fZ;
				float overallForce;
				float distanceSquared, distance;
				int n;

				x = pointsCoordinates[threadId * 3];
				y = pointsCoordinates[threadId * 3 + 1];
				z = pointsCoordinates[threadId * 3 + 2];

				for(n = 0; n < maxCells; n++)
				{
					if(activityFlag[n] == 1 && n != threadId)
					{
						nX = pointsCoordinates[n * 3];
						nY = pointsCoordinates[n * 3 + 1];
						nZ = pointsCoordinates[n * 3 + 2];

						dX = nX - x;
						dY = nY - y;
						dZ = nZ - z;

						if(dX != 0 || dY != 0 || dZ != 0)
						{
							distanceSquared = dX * dX + dY * dY + dZ * dZ;
							distance = sqrtf(distanceSquared);

							overallForce = -copysignf( repulsion, logf(distance/repulsionDistance)) / distanceSquared;


							//overallForce = ((distance > repulsionDistance) * ( -repulsion) + (distance <= repulsionDistance) * repulsion ) / distanceSquared;
							
							fX = overallForce * dX / distance;
							fY = overallForce * dY / distance;
							fZ = overallForce * dZ / distance;

							force[threadId * 3] += -fX;
							force[threadId * 3 + 1] += -fY;
							force[threadId * 3 + 2] += - fZ;
						}
					}
				}
			}
			
		
		}
	}

	__global__ void UseForceKernel(
		
		float *force,
		float forceFactor,
		float *pointsCoordinates,
		int maxCells
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < maxCells * 3)
		{
			pointsCoordinates[threadId] += forceFactor * force[threadId];
		}
	}


	__global__ void CenterOfGravityKernel(
		
		float *pointsCoordinates,
		float *centerOfGravity,
		int *activityFlag,
		int maxCells

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < 1)
		{
			float xSum = 0.00f, ySum = 0.00f, zSum = 0.00f;
			int livingCells = 0;
			for(int c = 0; c < maxCells; c++)
			{
				if(activityFlag[c] == 1)
				{
					xSum += pointsCoordinates[c * 3];
					ySum += pointsCoordinates[c * 3 + 1];
					zSum += pointsCoordinates[c * 3 + 2];

					livingCells++;
				} 
			}
			centerOfGravity[0] = xSum / (float)livingCells;
			centerOfGravity[1] = ySum / (float)livingCells;
			centerOfGravity[2] = zSum / (float)livingCells;
		}
	}


	// GRAPHICS PART ---------------------------
	// data preparation for the observer


	__global__ void ZeroTextureKernel(
		
		unsigned int *texture,
		int count
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;
		
		if(threadId < count)
		{
			texture[threadId] = 0;
		}
	}


	__global__ void CopyPointsCoordinatesKernel(
		
		float *pointsCoordinates,
		int *activityFlag,
		float xNonValid,
		float yNonValid,
		float zNonValid,
		float *dataVertex,
		int dataVertexOffset,
		int maxCells
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < maxCells)
		{
			float xToCopy = pointsCoordinates[threadId * 3];
			float yToCopy = pointsCoordinates[threadId * 3 + 1];
			float zToCopy = pointsCoordinates[threadId * 3 + 2];
			if(activityFlag[threadId] == 0)
			{
				xToCopy = xNonValid;
				yToCopy = yNonValid;
				zToCopy = zNonValid;
			}
			dataVertex[dataVertexOffset + threadId * 3] = xToCopy;
			dataVertex[dataVertexOffset + threadId * 3 + 1] = yToCopy;
			dataVertex[dataVertexOffset + threadId * 3 + 2] = zToCopy;
		}
	}

	__global__ void CopyConnectionsCoordinatesKernel(
		
		
		int *connectionMatrix,
		float *pointsCoordinates,
		float *vertexData,
		int *connectionCount,
		int maxCells

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < maxCells * maxCells)
		{
			if(connectionMatrix[threadId] == 1)
			{
				int from = threadId / maxCells;
				int to = threadId % maxCells;

				if(to > from)
				{
					//int vertexDataOffset = maxCells * 3;
					int vertexDataOffset = 0;
					int connIdx = atomicAdd( &connectionCount[0], 1);

					vertexData[vertexDataOffset + connIdx * 6] = pointsCoordinates[from * 3];
					vertexData[vertexDataOffset + connIdx * 6 + 1] = pointsCoordinates[from * 3 + 1];
					vertexData[vertexDataOffset + connIdx * 6 + 2] = pointsCoordinates[from * 3 + 2];

					vertexData[vertexDataOffset + connIdx * 6 + 3] = pointsCoordinates[to * 3];
					vertexData[vertexDataOffset + connIdx * 6 + 4] = pointsCoordinates[to * 3 + 1];
					vertexData[vertexDataOffset + connIdx * 6 + 5] = pointsCoordinates[to * 3 + 2];
				}
			
			}
			
		}
	}


	__global__ void ComputeQuadsKernel(
		
		float *pointsCoordinates,
		float *vertexData,
		int quadOffset,
		float textureSide,
		int *activityFlag,
		int textureWidth,
		int maxCells

		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < maxCells)
		{
			float x = pointsCoordinates[threadId * 3];
			float y = pointsCoordinates[threadId * 3 + 1];
			float z = pointsCoordinates[threadId * 3 + 2];

			float halfSide = 0.50f * textureSide;
			if(activityFlag[threadId] == 0)
			{
				halfSide = 0.00f;
			}

			int textureOffset = quadOffset + maxCells * 4 * 3 * 3;
			float textureAbsLength = (float)(maxCells * textureWidth);

			// vertical x-alligned
			vertexData[quadOffset + threadId * 36] = x - halfSide;
			vertexData[quadOffset + threadId * 36 + 1] = y + halfSide;
			vertexData[quadOffset + threadId * 36 + 2] = z;

			vertexData[textureOffset + threadId * 24] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 24 + 1] = 0.00f;

			vertexData[quadOffset + threadId * 36 + 3] = x - halfSide;
			vertexData[quadOffset + threadId * 36 + 4] = y - halfSide;
			vertexData[quadOffset + threadId * 36 + 5] = z;

			vertexData[textureOffset + threadId * 24 + 2] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 24 + 3] = 1.00f;
			
			vertexData[quadOffset + threadId * 36 + 6] = x + halfSide;
			vertexData[quadOffset + threadId * 36 + 7] = y - halfSide;
			vertexData[quadOffset + threadId * 36 + 8] = z;

			vertexData[textureOffset + threadId * 24 + 4] = (float)((threadId+1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 24 + 5] = 1.00f;


			vertexData[quadOffset + threadId * 36 + 9] = x + halfSide;
			vertexData[quadOffset + threadId * 36 + 10] = y + halfSide;
			vertexData[quadOffset + threadId * 36 + 11] = z;

			vertexData[textureOffset + threadId * 24 + 6] = (float)((threadId+1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 24 + 7] = 0.00f;

			// horizontal
			vertexData[quadOffset + threadId * 36 + 12] = x - halfSide;
			vertexData[quadOffset + threadId * 36 + 13] = y;
			vertexData[quadOffset + threadId * 36 + 14] = z + halfSide;

			vertexData[textureOffset + threadId * 24 + 8] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 24 + 9] = 1.00f;

			vertexData[quadOffset + threadId * 36 + 15] = x - halfSide;
			vertexData[quadOffset + threadId * 36 + 16] = y;
			vertexData[quadOffset + threadId * 36 + 17] = z - halfSide;

			vertexData[textureOffset + threadId * 24 + 10] = (float)(threadId * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 24 + 11] = 0.00f;

			vertexData[quadOffset + threadId * 36 + 18] = x + halfSide;
			vertexData[quadOffset + threadId * 36 + 19] = y;
			vertexData[quadOffset + threadId * 36 + 20] = z - halfSide;

			vertexData[textureOffset + threadId * 24 + 12] = (float)((threadId+1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 24 + 13] = 0.00f;

			vertexData[quadOffset + threadId * 36 + 21] = x + halfSide;
			vertexData[quadOffset + threadId * 36 + 22] = y;
			vertexData[quadOffset + threadId * 36 + 23] = z + halfSide;

			vertexData[textureOffset + threadId * 24 + 14] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 24 + 15] = 1.00f;

			// vertical z-alligned
			vertexData[quadOffset + threadId * 36 + 24] = x;
			vertexData[quadOffset + threadId * 36 + 25] = y - halfSide;
			vertexData[quadOffset + threadId * 36 + 26] = z + halfSide;

			vertexData[textureOffset + threadId * 24 + 16] = (float)((threadId+1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 24 + 17] = 1.00f;

			vertexData[quadOffset + threadId * 36 + 27] = x;
			vertexData[quadOffset + threadId * 36 + 28] = y - halfSide;
			vertexData[quadOffset + threadId * 36 + 29] = z - halfSide;

			vertexData[textureOffset + threadId * 24 + 18] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 24 + 19] = 1.00f;

			vertexData[quadOffset + threadId * 36 + 30] = x;
			vertexData[quadOffset + threadId * 36 + 31] = y + halfSide;
			vertexData[quadOffset + threadId * 36 + 32] = z - halfSide;

			vertexData[textureOffset + threadId * 24 + 20] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 24 + 21] = 0.00f;

			vertexData[quadOffset + threadId * 36 + 33] = x;
			vertexData[quadOffset + threadId * 36 + 34] = y + halfSide;
			vertexData[quadOffset + threadId * 36 + 35] = z + halfSide;

			vertexData[textureOffset + threadId * 24 + 22] = (float)((threadId+1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 24 + 23] = 0.00f;
		}
	}

	__constant__ float operationMaskConstant[72];
	__constant__ float cubeTexCoordinatesConstant[48];

	__global__ void CubeCoordinatesKernel(
		
		float *vertexData,
		float *cubeOperation,
		int quadOffset,
		int *activityFlag,
		float cubeSize,
		float *pointsCoordinates,
		int maxCells
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < maxCells * 72)
		{
			int cellId = threadId / 72;
			int sideId = (threadId / 12) % 6;
			int pointId = threadId % 12;
			int coordId = threadId % 3;

			float halfSide = (activityFlag[cellId] == 1) * 0.50f * cubeSize;
			
			int textureOffset = quadOffset + maxCells * 4 * 6 * 3;

			//vertexData[quadOffset + cellId * 72 + sideId * 12 + pointId] = pointsCoordinates[cellId * 3 + coordId] + cubeOperation[sideId * 12 + pointId] * halfSide;
			vertexData[quadOffset + cellId * 72 + sideId * 12 + pointId] = pointsCoordinates[cellId * 3 + coordId] + operationMaskConstant[sideId * 12 + pointId] * halfSide;
		}
	}


	__global__ void CubeTextureKernel(
		
		float *vertexData,
		int texCoorOffset,
		float *cubeTexCoordinates,
		float cubeSize,
		float textureWidth,
		int *activityFlag,
		int maxCells
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < maxCells * 48)
		{
			int cellId = threadId / 48;
			float fCellId = (float)cellId;
			int sideId = (threadId / 8) % 6;
			int pointId = threadId % 8;
			int coordId = threadId % 3;

			float textureAbsLength = (float)maxCells * textureWidth;

			float halfSide = (activityFlag[cellId] == 1) * 0.50f * cubeSize;
			
			//vertexData[texCoorOffset + cellId * 48 + sideId * 8 + pointId] = (pointId % 2 == 0 ) * (((fCellId + cubeTexCoordinates[sideId * 8 + pointId])* textureWidth) / textureAbsLength)  + (pointId % 2 == 1) * (cubeTexCoordinates[sideId * 8 + pointId]);
			vertexData[texCoorOffset + cellId * 48 + sideId * 8 + pointId] = (pointId % 2 == 0 ) * (((fCellId + cubeTexCoordinatesConstant[sideId * 8 + pointId])* textureWidth) / textureAbsLength)  + (pointId % 2 == 1) * (cubeTexCoordinatesConstant[sideId * 8 + pointId]);
			//vertexData[textureOffset + cellId * 48 + 8 * sideId]     = ((fCellId + cubeTexCoordinates[sideId * 8])* textureWidth) / textureAbsLength;
			//vertexData[textureOffset + cellId * 48 + 8 * sideId + 1] = cubeTexCoordinates[sideId * 8 + 1];
		}

	}

	
	__global__ void WinnersKernel(
		
		float *winner,
		float *vertexData,
		int vertexOffset,
		float *pointsCoordinates,
		float cubeSize,
		int maxCells
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < maxCells)
		{
			if(winner[threadId] == 1.00f)
			{
				float x = pointsCoordinates[threadId * 3];
				float y = pointsCoordinates[threadId * 3 + 1];
				float z = pointsCoordinates[threadId * 3 + 2];

				float side = 1.2f * cubeSize;
				float halfSize = 0.50f * side;

				// bottom side
				vertexData[vertexOffset] = x - halfSize;
				vertexData[vertexOffset + 1] = y - halfSize;
				vertexData[vertexOffset + 2] = z - halfSize;

				vertexData[vertexOffset + 3] = x - halfSize;
				vertexData[vertexOffset + 4] = y - halfSize;
				vertexData[vertexOffset + 5] = z + halfSize;


				vertexData[vertexOffset + 6] = x + halfSize;
				vertexData[vertexOffset + 7] = y - halfSize;
				vertexData[vertexOffset + 8] = z + halfSize;

				vertexData[vertexOffset + 9] = x + halfSize;
				vertexData[vertexOffset + 10] = y - halfSize;
				vertexData[vertexOffset + 11] = z - halfSize;
			}
		}
	}







	__global__ void ComputeCubes2Kernel(
		
		float *pointsCoordinates,
		float *vertexData,
		int quadOffset,
		float cubeSide,
		float *cubeOperation,
		float *cubeTexCoordinates,
		int *activityFlag,
		float textureWidth,
		int maxCells
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < maxCells * 6)
		{
			int cellId = threadId / 6;
			float fCellId = (float)cellId;
			int sideId = threadId % 6;

			float x = pointsCoordinates[cellId * 3];
			float y = pointsCoordinates[cellId * 3 + 1];
			float z = pointsCoordinates[cellId * 3 + 2];

			float halfSide = (activityFlag[cellId] == 1) * 0.50f * cubeSide;
			
			int textureOffset = quadOffset + maxCells * 4 * 6 * 3;
			float textureAbsLength = (float)maxCells * textureWidth;


			vertexData[quadOffset + cellId * 72 + 12*sideId]     = x + operationMaskConstant[12*sideId] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 1] = y + operationMaskConstant[12*sideId + 1] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 2] = z + operationMaskConstant[12*sideId + 2] * halfSide;
			
			vertexData[quadOffset + cellId * 72 + 12*sideId + 3] = x + operationMaskConstant[12*sideId + 3] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 4] = y + operationMaskConstant[12*sideId + 4] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 5] = z + operationMaskConstant[12*sideId + 5] * halfSide;

			vertexData[quadOffset + cellId * 72 + 12*sideId + 6] = x + operationMaskConstant[12*sideId + 6] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 7] = y + operationMaskConstant[12*sideId + 7] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 8] = z + operationMaskConstant[12*sideId + 8] * halfSide;

			vertexData[quadOffset + cellId * 72 + 12*sideId + 9]  = x + operationMaskConstant[12*sideId + 9] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 10] = y + operationMaskConstant[12*sideId + 10] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 11] = z + operationMaskConstant[12*sideId + 11] * halfSide;


			vertexData[textureOffset + cellId * 48 + 8 * sideId]     = ((fCellId + cubeTexCoordinatesConstant[sideId * 8])* textureWidth) / textureAbsLength;
			vertexData[textureOffset + cellId * 48 + 8 * sideId + 1] = cubeTexCoordinatesConstant[sideId * 8 + 1];
			
			vertexData[textureOffset + cellId * 48 + 8 * sideId + 2] = ((fCellId + cubeTexCoordinatesConstant[sideId * 8 + 2]) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + cellId * 48 + 8 * sideId + 3] = cubeTexCoordinatesConstant[sideId * 8 + 3];

			vertexData[textureOffset + cellId * 48 + 8 * sideId + 4] = ((fCellId + cubeTexCoordinatesConstant[sideId * 8 + 4]) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + cellId * 48 + 8 * sideId + 5] = cubeTexCoordinatesConstant[sideId * 8 + 5];

			vertexData[textureOffset + cellId * 48 + 8 * sideId + 6] = ((fCellId + cubeTexCoordinatesConstant[sideId * 8 + 6]) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + cellId * 48 + 8 * sideId + 7] = cubeTexCoordinatesConstant[sideId * 8 + 7];


			/*
			vertexData[quadOffset + cellId * 72 + 12*sideId]     = x + cubeOperation[12*sideId] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 1] = y + cubeOperation[12*sideId + 1] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 2] = z + cubeOperation[12*sideId + 2] * halfSide;
			
			vertexData[quadOffset + cellId * 72 + 12*sideId + 3] = x + cubeOperation[12*sideId + 3] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 4] = y + cubeOperation[12*sideId + 4] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 5] = z + cubeOperation[12*sideId + 5] * halfSide;

			vertexData[quadOffset + cellId * 72 + 12*sideId + 6] = x + cubeOperation[12*sideId + 6] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 7] = y + cubeOperation[12*sideId + 7] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 8] = z + cubeOperation[12*sideId + 8] * halfSide;

			vertexData[quadOffset + cellId * 72 + 12*sideId + 9]  = x + cubeOperation[12*sideId + 9] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 10] = y + cubeOperation[12*sideId + 10] * halfSide;
			vertexData[quadOffset + cellId * 72 + 12*sideId + 11] = z + cubeOperation[12*sideId + 11] * halfSide;


			vertexData[textureOffset + cellId * 48 + 8 * sideId]     = ((fCellId + cubeTexCoordinates[sideId * 8])* textureWidth) / textureAbsLength;
			vertexData[textureOffset + cellId * 48 + 8 * sideId + 1] = cubeTexCoordinates[sideId * 8 + 1];
			
			vertexData[textureOffset + cellId * 48 + 8 * sideId + 2] = ((fCellId + cubeTexCoordinates[sideId * 8 + 2]) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + cellId * 48 + 8 * sideId + 3] = cubeTexCoordinates[sideId * 8 + 3];

			vertexData[textureOffset + cellId * 48 + 8 * sideId + 4] = ((fCellId + cubeTexCoordinates[sideId * 8 + 4]) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + cellId * 48 + 8 * sideId + 5] = cubeTexCoordinates[sideId * 8 + 5];

			vertexData[textureOffset + cellId * 48 + 8 * sideId + 6] = ((fCellId + cubeTexCoordinates[sideId * 8 + 6]) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + cellId * 48 + 8 * sideId + 7] = cubeTexCoordinates[sideId * 8 + 7];
			*/
		}
	}


	__global__ void ComputeCubesKernel(
	
		float *pointsCoordinates,
		float *vertexData,
		int quadOffset,
		float cubeSide,
		int *activityFlag,
		int textureWidth,
		int maxCells

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < maxCells)
		{
			float x = pointsCoordinates[threadId * 3];
			float y = pointsCoordinates[threadId * 3 + 1];
			float z = pointsCoordinates[threadId * 3 + 2];

			float halfSide = 0.50f * cubeSide;
			if(activityFlag[threadId] == 0)
			{
				halfSide = 0.00f;
			}

			int textureOffset = quadOffset + maxCells * 4 * 6 * 3;
			float textureAbsLength = (float)(maxCells * textureWidth);


			// BOTTOM SIDE
			vertexData[quadOffset + threadId * 72] = x - halfSide;
			vertexData[quadOffset + threadId * 72 + 1] = y - halfSide;
			vertexData[quadOffset + threadId * 72 + 2] = z + halfSide;
			
			vertexData[quadOffset + threadId * 72 + 3] = x - halfSide;
			vertexData[quadOffset + threadId * 72 + 4] = y - halfSide;
			vertexData[quadOffset + threadId * 72 + 5] = z - halfSide;

			vertexData[quadOffset + threadId * 72 + 6] = x + halfSide;
			vertexData[quadOffset + threadId * 72 + 7] = y - halfSide;
			vertexData[quadOffset + threadId * 72 + 8] = z - halfSide;

			vertexData[quadOffset + threadId * 72 + 9] = x + halfSide;
			vertexData[quadOffset + threadId * 72 + 10] = y - halfSide;
			vertexData[quadOffset + threadId * 72 + 11] = z + halfSide;

			vertexData[textureOffset + threadId * 48] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 1] = 0.00f;
			
			vertexData[textureOffset + threadId * 48 + 2] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 3] = 1.00f;

			vertexData[textureOffset + threadId * 48 + 4] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 5] = 1.00f;

			vertexData[textureOffset + threadId * 48 + 6] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 7] = 0.00f;


			// FRONT SIDE
			vertexData[quadOffset + threadId * 72 + 12] = x - halfSide;
			vertexData[quadOffset + threadId * 72 + 13] = y + halfSide;
			vertexData[quadOffset + threadId * 72 + 14] = z + halfSide;
			
			vertexData[quadOffset + threadId * 72 + 15] = x - halfSide;
			vertexData[quadOffset + threadId * 72 + 16] = y - halfSide;
			vertexData[quadOffset + threadId * 72 + 17] = z + halfSide;

			vertexData[quadOffset + threadId * 72 + 18] = x + halfSide;
			vertexData[quadOffset + threadId * 72 + 19] = y - halfSide;
			vertexData[quadOffset + threadId * 72 + 20] = z + halfSide;

			vertexData[quadOffset + threadId * 72 + 21] = x + halfSide;
			vertexData[quadOffset + threadId * 72 + 22] = y + halfSide;
			vertexData[quadOffset + threadId * 72 + 23] = z + halfSide;

			
			
			vertexData[textureOffset + threadId * 48 + 8] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 9] = 0.00f;

			vertexData[textureOffset + threadId * 48 + 10] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 11] = 1.00f;

			vertexData[textureOffset + threadId * 48 + 12] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 13] = 1.00f;

			vertexData[textureOffset + threadId * 48 + 14] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 15] = 0.00f;

			
			
			// LEFT SIDE
			vertexData[quadOffset + threadId * 72 + 24] = x - halfSide;
			vertexData[quadOffset + threadId * 72 + 25] = y + halfSide;
			vertexData[quadOffset + threadId * 72 + 26] = z - halfSide;
			
			vertexData[quadOffset + threadId * 72 + 27] = x - halfSide;
			vertexData[quadOffset + threadId * 72 + 28] = y - halfSide;
			vertexData[quadOffset + threadId * 72 + 29] = z - halfSide;

			vertexData[quadOffset + threadId * 72 + 30] = x - halfSide;
			vertexData[quadOffset + threadId * 72 + 31] = y - halfSide;
			vertexData[quadOffset + threadId * 72 + 32] = z + halfSide;

			vertexData[quadOffset + threadId * 72 + 33] = x - halfSide;
			vertexData[quadOffset + threadId * 72 + 34] = y + halfSide;
			vertexData[quadOffset + threadId * 72 + 35] = z + halfSide;


			vertexData[textureOffset + threadId * 48 + 16] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 17] = 0.00f;

			vertexData[textureOffset + threadId * 48 + 18] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 19] = 1.00f;

			vertexData[textureOffset + threadId * 48 + 20] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 21] = 1.00f;

			vertexData[textureOffset + threadId * 48 + 22] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 23] = 0.00f;

			// BACK SIDE
			vertexData[quadOffset + threadId * 72 + 36] = x - halfSide;
			vertexData[quadOffset + threadId * 72 + 37] = y + halfSide;
			vertexData[quadOffset + threadId * 72 + 38] = z - halfSide;
			
			vertexData[quadOffset + threadId * 72 + 39] = x - halfSide;
			vertexData[quadOffset + threadId * 72 + 40] = y - halfSide;
			vertexData[quadOffset + threadId * 72 + 41] = z - halfSide;

			vertexData[quadOffset + threadId * 72 + 42] = x + halfSide;
			vertexData[quadOffset + threadId * 72 + 43] = y - halfSide;
			vertexData[quadOffset + threadId * 72 + 44] = z - halfSide;

			vertexData[quadOffset + threadId * 72 + 45] = x + halfSide;
			vertexData[quadOffset + threadId * 72 + 46] = y + halfSide;
			vertexData[quadOffset + threadId * 72 + 47] = z - halfSide;


			vertexData[textureOffset + threadId * 48 + 24] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 25] = 0.00f;

			vertexData[textureOffset + threadId * 48 + 26] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 27] = 1.00f;

			vertexData[textureOffset + threadId * 48 + 28] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 29] = 1.00f;

			vertexData[textureOffset + threadId * 48 + 30] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 31] = 0.00f;


			// RIGHT SIDE
			vertexData[quadOffset + threadId * 72 + 48] = x + halfSide;
			vertexData[quadOffset + threadId * 72 + 49] = y + halfSide;
			vertexData[quadOffset + threadId * 72 + 50] = z - halfSide;
			
			vertexData[quadOffset + threadId * 72 + 51] = x + halfSide;
			vertexData[quadOffset + threadId * 72 + 52] = y - halfSide;
			vertexData[quadOffset + threadId * 72 + 53] = z - halfSide;

			vertexData[quadOffset + threadId * 72 + 54] = x + halfSide;
			vertexData[quadOffset + threadId * 72 + 55] = y - halfSide;
			vertexData[quadOffset + threadId * 72 + 56] = z + halfSide;

			vertexData[quadOffset + threadId * 72 + 57] = x + halfSide;
			vertexData[quadOffset + threadId * 72 + 58] = y + halfSide;
			vertexData[quadOffset + threadId * 72 + 59] = z + halfSide;

			vertexData[textureOffset + threadId * 48 + 32] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 33] = 0.00f;
			
			vertexData[textureOffset + threadId * 48 + 34] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 35] = 1.00f;

			vertexData[textureOffset + threadId * 48 + 36] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 37] = 1.00f;

			vertexData[textureOffset + threadId * 48 + 38] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 39] = 0.00f;


			// UPPER SIDE
			vertexData[quadOffset + threadId * 72 + 60] = x - halfSide;
			vertexData[quadOffset + threadId * 72 + 61] = y + halfSide;
			vertexData[quadOffset + threadId * 72 + 62] = z + halfSide;
			
			vertexData[quadOffset + threadId * 72 + 63] = x - halfSide;
			vertexData[quadOffset + threadId * 72 + 64] = y + halfSide;
			vertexData[quadOffset + threadId * 72 + 65] = z - halfSide;

			vertexData[quadOffset + threadId * 72 + 66] = x + halfSide;
			vertexData[quadOffset + threadId * 72 + 67] = y + halfSide;
			vertexData[quadOffset + threadId * 72 + 68] = z - halfSide;

			vertexData[quadOffset + threadId * 72 + 69] = x + halfSide;
			vertexData[quadOffset + threadId * 72 + 70] = y + halfSide;
			vertexData[quadOffset + threadId * 72 + 71] = z + halfSide;


			
			vertexData[textureOffset + threadId * 48 + 40] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 41] = 1.00f;

			vertexData[textureOffset + threadId * 48 + 42] = (float)((threadId) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 43] = 0.00f;

			vertexData[textureOffset + threadId * 48 + 44] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 45] = 0.00f;

			vertexData[textureOffset + threadId * 48 + 46] = (float)((threadId + 1) * textureWidth) / textureAbsLength;
			vertexData[textureOffset + threadId * 48 + 47] = 1.00f;
		}


	
	}




	__global__ void CopyAndProcessTextureKernel(
		
		float *referenceVector,
		int referenceVectorSize,
		int textureWidth,
		int textureFieldWidth,
		unsigned int *pixels,
		int maxCells,
		int count

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < count)
		{
			int cellId = threadId / referenceVectorSize;
			int pixelXPos = (threadId - cellId * referenceVectorSize) % textureWidth;
			int pixelYPos = (threadId - cellId * referenceVectorSize) / textureWidth;

			float hue = 1;
			float saturation = 0;
			float value = fminf(1, fmaxf(-1, (referenceVector[threadId] + 1) * 0.50f));

			pixels[pixelYPos * textureFieldWidth +  cellId * textureWidth + pixelXPos] = hsva_to_uint_rgba(hue, saturation ,value, 1.00f);
			//pixels[pixelYPos * textureFieldWidth +  cellId * textureWidth + pixelXPos] = 0xFF000000 + (int)(referenceVector[threadId] * 255.00f);
		}
	}

}
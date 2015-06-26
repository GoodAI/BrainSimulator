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

	//kernel code
	__global__ void VectorInputDiffKernel(
		
		float *input,
		int inputSize,
		float *referenceVector,
		int maxCells,
		float *difference
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < maxCells * inputSize)
		{
			difference[threadId] = input[threadId % inputSize] - referenceVector[threadId];
		}
	}


	__global__ void ComputeDistanceKernel(
		
		int inputSize,
		float *distance,
		float *dimensionWeight,
		int maxCells,
		float *difference
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < maxCells)
		{
			float sum = 0.00f;
			float value;
			for(int i = 0; i < inputSize; i++)
			{
				value = difference[threadId * inputSize + i];
				sum += dimensionWeight[i] * value*value;
			}
			distance[threadId] = sqrtf(sum);
		}
	}


	__global__ void AddLocalErrorKernel(
		
		int s1,
		float *distance,
		float *localError

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < 1)
		{
			localError[s1] += distance[s1] * distance[s1];
		}

	}

	__global__ void AddUtilityKernel(
		
		int s1,
		int s2,
		float *distance,
		float *utility
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < 1)
		{
			utility[s1] += distance[s2] - distance[s1];
		}
	}


	__global__ void AdaptWinningFractionKernel(
			
		int s1,
		float *winningFraction,
		int *winningCount,
		float bParam,
		int maxCells

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < maxCells)
		{
			winningFraction[threadId] = winningFraction[threadId] + bParam * ((float)(threadId == s1) - winningFraction[threadId]);
			winningCount[threadId] = winningCount[threadId] + (threadId == s1) * 1;
		}
	}

	__global__ void ComputeBiasTermKernel(
			
		float *biasTerm,
		float cFactor,
		float *winningFraction,
		int activeCells,
		int maxCells

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < maxCells)
		{
			biasTerm[threadId] = cFactor * ( 1.00f / activeCells - winningFraction[threadId]);
		}
	}

	__global__ void ComputeBiasedDistanceKernel(
		
		float *distance,
		float *biasedDistance,
		float *biasTerm,
		int maxCells

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < maxCells)
		{
			biasedDistance[threadId] = distance[threadId] + biasTerm[threadId];
		}
	}

	__global__ void CreateAndRefreshConnectionKernel(

		int s1,
		int s2,
		int *connection,
		int *age,
		int maxCells

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < 1)
		{
			connection[s1 * maxCells + s2] = 1;
			age[s1 * maxCells + s2] = 0;
			connection[s2 * maxCells + s1] = 1;
			age[s2 * maxCells + s1] = 0;
		}
	}


	__global__ void AdaptRefVectorKernel(
		
		int cell,
		float *referenceVector,
		float oldErrorFraction,
		float youngErrorFraction,
		float decayFactor,
		int *winningCount,
		float *difference,
		int inputSize
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < inputSize)
		{
			float errorFraction = (youngErrorFraction - oldErrorFraction) * expf( - decayFactor * winningCount[cell] ) + oldErrorFraction;
			referenceVector[cell * inputSize + threadId] += errorFraction * difference[cell * inputSize + threadId];
		}
	}


	__global__ void IncrementConnectionAgeKernel(
		
		int cell,
		int *connection,
		int *age,
		int maxCells
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < maxCells)
		{
			if(connection[cell * maxCells + threadId] == 1)
			{
				age[cell * maxCells + threadId] += 1;
				age[threadId * maxCells + cell] += 1;
			}
			
		}
	}

	__global__ void RemoveEdgesKernel(

		int *connection,
		int *age,
		int maxAge,
		int *activityFlag,
		float *winningFraction,
		int *winningCount,
		float *utility,
		float *localError,
		int *neuronAge,
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
				neuronAge[threadId] = neuronAge[threadId] + 1;

				// TO DO : GET RID OF  IFs & ELSEs
				int activeConnections = 0;
				int connId;
				for(int c = 0; c < maxCells; c++)
				{
					connId = threadId * maxCells + c;
					if(connection[connId] == 1)
					{
						if(age[connId] <= maxAge)
						{
							activeConnections++;
						}
						else
						{
							connection[connId] = 0;
							age[connId] = 0;
						}
					}
				}
				if(activeConnections == 0)
				{
					activityFlag[threadId] = 0;
					localError[threadId] = 0.00f;
					neuronAge[threadId] = 0;
					winningFraction[threadId] = 0.00f;
					winningCount[threadId] = 0;
					utility[threadId] = 0.00f;
				}
			}
		}
	}

	__global__ void RemoveNodeByUtilityKernel(
		
		int *connectionMatrix,
		int *connectionAge,
		int *activityFlag,
		float *utility,
		float utilityConstant,
		float *localError,
		int *neuronAge,
		float *winningFraction,
		int *winningCount,
		float maxError,
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
				if(utility[threadId] > 0.00f)
				{
					if( maxError / utility[threadId] > utilityConstant )
					{
						activityFlag[threadId] = 0;
						localError[threadId] = 0.00f;
						neuronAge[threadId] = 0;
						winningFraction[threadId] = 0.00f;
						winningCount[threadId] = 0;
						utility[threadId] = 0.00f;

						for(int n = 0; n < maxCells; n++)
						{
							connectionMatrix[threadId * maxCells + n] = 0;
							connectionAge[threadId * maxCells + n] = 0;
							connectionMatrix[n * maxCells + threadId] = 0;
							connectionAge[n * maxCells + threadId] = 0;
						}
					}
				}
			}
		}
	}

	__global__ void InterpolateVectorKernel(
	
		int r,
		int q,
		int f,
		int inputSize,
		float *referenceVector
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < inputSize)
		{
			referenceVector[r * inputSize + threadId] = 0.50f * (referenceVector[q * inputSize + threadId] + referenceVector[f * inputSize + threadId]);
		}
	}

	__global__ void NewNodeConnectionKernel(
		
		int f,
		int q,
		int r,
		int *activityFlag,
		int *connection,
		int *age,
		float *localError,
		float alfa,
		int maxCells,
		float errorFraction
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < 1)
		{
			activityFlag[r] = 1;

			connection[q * maxCells + f] = 0;
			age[q * maxCells + f] = 0;
			connection[f * maxCells + q] = 0;
			age[f * maxCells + q] = 0;
			connection[q * maxCells + r] = 1;
			age[q * maxCells + r] = 0;
			connection[r * maxCells + q] = 1;
			age[r * maxCells + q] = 0;
			connection[f * maxCells + r] = 1;
			age[f * maxCells + r] = 0;
			connection[r * maxCells + f] = 1;
			age[r * maxCells + f] = 0;

			localError[q] -= alfa * localError[q];
			localError[f] -= alfa * localError[f];

			localError[r] = errorFraction * (localError[q] + localError[f]);
		}
	}


	__global__ void AddAndRefreshConnectionKernel(
		
		int node1,
		int node2,
		int *activityFlag,
		int *connection,
		int *age,
		int maxCells
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < 1)
		{
			activityFlag[node1] = 1;
			activityFlag[node2] = 1;

			connection[node1 * maxCells + node2] = 1;
			age[node1 * maxCells + node2] = 0;
			connection[node2 * maxCells + node1] = 1;
			age[node2 * maxCells + node1] = 0;
		}
	}

	__global__ void TwoNodesDifferenceKernel(
		
		int nodeOne,
		int nodeTwo,
		int vectorLength,
		float *referenceVector,
		float *twoNodesDifference

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < vectorLength)
		{
			twoNodesDifference[threadId] = referenceVector[nodeOne * vectorLength + threadId] - referenceVector[nodeTwo * vectorLength + threadId];
		}
	}

	__global__ void TwoNodesDistanceKernel(
		
		float *twoNodesDifference,
		float *twoNodesDistance,
		int vectorLength

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < 1)
		{
			float sum = 0.00f;
			float value;
			for(int i = 0; i < vectorLength; i++)
			{
				value = twoNodesDifference[threadId * vectorLength + i];
				sum += value*value;
			}
			twoNodesDistance[threadId] = sqrtf(sum);
		}
	}

	__global__ void CopyVectorKernel(
		
		float *from,
		int fromOffset,
		float *to,
		int toOffset,
		int vectorSize

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < vectorSize)
		{
			to[threadId + toOffset] = from[threadId + fromOffset];
		}
		
	}

	__global__ void DecreaseErrorAndUtilityKernel(
		
		float *localError,
		float *utility,
		int *activityFlag,
		int maxCells,
		float beta

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < maxCells)
		{
			if(activityFlag[threadId] == 1)
			{
				localError[threadId] -= beta * localError[threadId];
				utility[threadId] -= beta * utility[threadId];
			}
		}
	}

	__global__ void ComputeErrorPerWinningKernel(
		
		float *localError,
		int *winningCount,
		float *errorPerWinning,
		int *activityFlag,
		int maxCells
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;


		// TO DO: GET RID OF IF-ELSE
		if(threadId < maxCells)
		{
			if(activityFlag[threadId] == 1)
			{
				if(winningCount[threadId] != 0)
				{
					errorPerWinning[threadId] = localError[threadId] / (float)winningCount[threadId];
				}
				else
				{
					errorPerWinning[threadId] = 0.00f;
				}
			}
		}
	}
}
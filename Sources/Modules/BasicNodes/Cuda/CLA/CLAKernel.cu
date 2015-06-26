#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math_constants.h>


extern "C"  
{

	//kernel code
	__global__ void ComputeOverlapKernel(
		float *input,
		int *synapses,
		float *permanence,
		float *overlap,
		float *boost,
		float permanenceThreshold,
		float overlapThreshold,
		int synPerColumn,
		int columns
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < columns)
		{
			int from, synId;
			float overlapCounter = 0.00f;
			// for each input synapse
			for(int i = 0; i < synPerColumn; i++)
			{
				synId = threadId * synPerColumn + i;
				from = synapses[synId];
				if(from != -1)
				{
					// if the input is one and the permanence is bigger than threshold
					overlapCounter = overlapCounter + (float)(input[from] == 1.00f) * (permanence[synId] >= permanenceThreshold);
				}
			}
			overlap[threadId] = boost[threadId] * overlapCounter * (overlapCounter >= overlapThreshold);
		}
	}

	__global__ void ComputeRealOverlapKernel(
		// input
		float *input,
		// connections
		int *synapses,
		float *permanence,
		// columns
		float *overlap,
		float *boost,
		// constants
		int synPerColumn,
		int columns
		)
	{
		int columnId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
	
		if(columnId < columns)
		{
			int fromId, synId;
			float overlapCounter = 0.00f;
			float difference;

			for(int s = 0; s < synPerColumn; s++)
			{
				synId = columnId * synPerColumn + s;
				fromId = synapses[synId];
				if(fromId != -1)
				{
					difference = permanence[synId] - input[fromId];
					overlapCounter += difference * difference;
				}
			}
			overlap[columnId] = - overlapCounter / boost[columnId];
		}
	}


	__global__ void OverlapStatsKernel(
		float *overlap,
		int *overlapDutyCycle,
		int *overlapDutyCycleTrack,
		int stepInCycle,
		int columns
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < columns)
		{
			// subtract old value
			overlapDutyCycle[threadId] = overlapDutyCycle[threadId] - overlapDutyCycleTrack[stepInCycle * columns + threadId];
			
			// get actual overlap and convert it to integer (0 or 1)
			int actualBinaryOverlap = 1 * (overlap[threadId] > 0.00f);

			// add actual value
			overlapDutyCycle[threadId] = overlapDutyCycle[threadId] + actualBinaryOverlap;

			// save actual value to the track field
			overlapDutyCycleTrack[stepInCycle * columns + threadId] = actualBinaryOverlap;
		}
	}

	__global__ void ActivityStatsKernel(
		
		int *activity,
		int *activityDutyCycle,
		int *activityDutyCycleTrack,
		int stepInCycle,
		int columns

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < columns)
		{
			// subtract old value
			activityDutyCycle[threadId] = activityDutyCycle[threadId] - activityDutyCycleTrack[stepInCycle * columns + threadId];

			// get actual value
			int actualActivity = activity[threadId];

			// add actual value
			activityDutyCycle[threadId] = activityDutyCycle[threadId] + actualActivity;

			// save actual value to the track field
			activityDutyCycleTrack[stepInCycle * columns + threadId] = actualActivity;

		}
	}


	__global__ void FindMinDutyCycleKernel(
		
		int *activeDutyCycle,
		float *minActiveDutyCycle,
		int regionSize,
		int neigborsRadius,
		int columns
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
					+ blockDim.x*blockIdx.x				//blocks preceeding current block
					+ threadIdx.x;

		if(threadId < columns)
		{
			int maxActiveDutyCycle = 0;
			int actualActiveDutyCycle;
			int neighbors = neigborsRadius * neigborsRadius - 1;

			int columnX = threadId % regionSize;
			int columnY = threadId / regionSize;

			int xCounter, yCounter;
			int neighborId;

			for(xCounter = columnX - neigborsRadius; xCounter <= columnX + neigborsRadius; xCounter++)
			{
				for(yCounter = columnY - neigborsRadius; yCounter <= columnY + neigborsRadius; yCounter++)
				{
					if(xCounter > 0 && xCounter < regionSize && yCounter > 0 && yCounter < regionSize && (xCounter != columnX && yCounter != columnY))
					{
						neighborId = yCounter * regionSize + xCounter;
						actualActiveDutyCycle = activeDutyCycle[neighborId];
						if(actualActiveDutyCycle > maxActiveDutyCycle)
						{
							maxActiveDutyCycle = actualActiveDutyCycle;
						}
					}
				}
			}

			minActiveDutyCycle[threadId] = 0.01f * (float)maxActiveDutyCycle;
		}
	}

	__global__ void BoostKernel(

		float *boost,
		int *activeDutyCycle,
		float minDutyCycle,
		int cycle,
		int columns
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < columns)
		{
			float floatActiveDutyCycle = (float)activeDutyCycle[threadId];
			boost[threadId] = 1.00f + (floatActiveDutyCycle < minDutyCycle) * (minDutyCycle - floatActiveDutyCycle);
		}
	}

	__global__ void LocalBoostKernel(
		
		float *boost,
		int *activeDutyCycle,
		float *minDutyCycle,
		int cycle,
		int columns
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < columns)
		{
			float floatActiveDutyCycle = (float)activeDutyCycle[threadId];
			boost[threadId] = 1.00f + (floatActiveDutyCycle < minDutyCycle[threadId]) * (minDutyCycle[threadId] - floatActiveDutyCycle);
		}
	}
	
	__global__ void IncreasePermanencesKernel(
		
		int *overlapDutyCycle,
		float minDutyCycle,
		float *permanences,
		float permanenceThreshold,
		float permanenceBoost,
		float maxPermanence,
		int synPerColumn,
		int columns

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < columns)
		{
			if((float)overlapDutyCycle[threadId] < minDutyCycle)
			{
				float actualPermanence;
				int synId;
				for(int s = 0; s < synPerColumn; s++)
				{
					synId = threadId * synPerColumn + s;
					actualPermanence = permanences[synId];
					permanences[synId] = fminf(maxPermanence, actualPermanence + permanenceBoost * permanenceThreshold);
				}
			}
		}
	}
	

	__global__ void LocalIncreasePermanencesKernel(
		
		int *overlapDutyCycle,
		float *minDutyCycle,
		float *permanences,
		float permanenceThreshold,
		float permanenceBoost,
		float maxPermanence,
		int synPerColumn,
		int columns
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < columns)
		{
			if((float)overlapDutyCycle[threadId] < minDutyCycle[threadId])
			{
				float actualPermanence;
				int synId;
				for(int s = 0; s < synPerColumn; s++)
				{
					synId = threadId * synPerColumn + s;
					actualPermanence = permanences[synId];
					permanences[synId] = fminf(maxPermanence, actualPermanence + permanenceBoost * permanenceThreshold);
				}
			}
		}
	}

	__global__ void ReconstructInputKernel(
		int *activity,
		int *synapses,
		int synPerColumn,
		float *permanences,
		float permanenceThreshold,
		float *canvas,
		int columns
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < columns)
		{
			if(activity[threadId] == 1)
			{
				int synId;
				int to;

				for(int s = 0; s < synPerColumn; s++)
				{
					synId = threadId * synPerColumn + s;
					to = synapses[synId];
					if(to != -1)
					{
						canvas[to] = 1 * (permanences[synId] >= permanenceThreshold);
					}
				}
			}
		}
	}

	__global__ void ReconstructRealInputKernel(
		int *activity,
		int *synapses,
		int synPerColumn,
		float *permanence,
		float *canvas,
		int *canvasCount,
		int columns
		)
	{
		int columnId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(columnId < columns)
		{
			if(activity[columnId] == 1)
			{
				int synId;
				int to;

				for(int s = 0; s < synPerColumn; s++)
				{
					synId = columnId * synPerColumn + s;
					to = synapses[synId];
					if(to != -1)
					{
						//canvas[to] = permanence[synId];
						atomicAdd(&canvas[to], permanence[synId]);
						atomicAdd(&canvasCount[to], 1);
					}
				}
			}
		}
	}

	__global__ void MeanReconstructionKernel(
		float *canvas,
		int *canvasCount,
		int count
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < count)
		{
			if(canvasCount[threadId] > 0)
			{
				canvas[threadId] = canvas[threadId] / (float)canvasCount[threadId];
			}
		}
	}

	//atomicAdd(&cellPreviousActiveCount[0], 1);

	__global__ void ShowColumnCenterKernel(
		
		int columnId,
		int *centerX,
		int *centerY,
		int columnHint,
		float *canvas
		
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < 1)
		{
			int x,y;
			x = centerX[columnId];
			y = centerY[columnId];

			//canvas[columnHint * y + x] = 1;
			canvas[x] = 1.00f;
		}
	}

	__global__ void ShowReceptiveFieldKernel(
		
		int columnId,
		int *synapses,
		float *permanence,
		float permanenceThreshold,
		int *centerX,
		int *centerY,
		int inputWidth,
		float *canvas,
		int synPerColumn

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < synPerColumn)
		{
			if(threadId == 0)
			{
				canvas[centerY[columnId] * inputWidth + centerX[columnId]] = CUDART_NAN_F;
			}
			__syncthreads();

			int synId = columnId * synPerColumn + threadId;
			int from = synapses[synId];
			if(from != -1)
			{
				canvas[from] = permanence[synId] - permanenceThreshold;
			}

		}

	}

	__global__ void AdaptPermanenceKernel(
		
		int *activity,
		int *synapses,
		float *input,
		float *permanence,
		float perIncrease,
		float perDecrease,
		int synPerColumn,
		int synapseCount

		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < synapseCount)
		{
			int columnId = threadId / synPerColumn;
			if(activity[columnId] == 1)
			{
				int inputId = synapses[threadId];
				if(inputId != -1)
				{
					float change = (input[inputId] == 1.00f) * (perIncrease) + (input[inputId] != 1.00f) * (-perDecrease);
					permanence[threadId] = permanence[threadId] + change;
				}
			}
		}
	}

	__global__ void AdaptRealPermanenceKernel(
		
		int *activity,
		int *synapses,
		float *input,
		float *permanence,
		float learningRate,
		int synPerColumn,
		int synapseCount
		)
	{
		int synapseId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(synapseId < synapseCount)
		{
			int columnId = synapseId / synPerColumn;
			if(activity[columnId] == 1)
			{
				int inputId = synapses[synapseId];
				if(inputId != -1)
				{
					float change = learningRate * (input[inputId] - permanence[synapseId]);
					permanence[synapseId] += change;
				}
			}
		}
	}

	__global__ void SaturatePermanenceKernel(
		float *permanence,
		float minPermanence,
		float maxPermanence,
		int synapseCount
		)
	{
		int synapseId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(synapseId < synapseCount)
		{
			permanence[synapseId] = fmaxf(fminf(permanence[synapseId], maxPermanence), minPermanence);
		}
	}

	__global__ void SendActivityToOutputKernel(
		int *activity,
		float *output,
		int columns
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < columns)
		{
			output[threadId] = (float)activity[threadId];
		}
	}


	


}
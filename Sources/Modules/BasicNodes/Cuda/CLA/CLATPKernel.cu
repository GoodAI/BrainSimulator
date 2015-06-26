#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>
#include <math_constants.h>


extern "C"  
{
	__global__ void PhaseOneInferenceKernel(
		// columns
		int *columnActivity,
		// cells
		int *cellActivity,
		int *cellPrediction,
		int *cellActiveSegment,
		// segment
		int *segmentSequenceFlag,
		// constants
		int cellsPerColumn,
		int columns
		)
	{
		int columnId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
	
		if(columnId < columns)
		{
			// for all active columns
			if(columnActivity[columnId] == 1)
			{
				int cellId;
				// bottom up prediction
				int buPredicted = 0;
				// find predicted cells
				for(int c = 0; c < cellsPerColumn; c++)
				{
					cellId = columnId * cellsPerColumn + c;
					// if the cell is in the predictive state
					if(cellPrediction[cellId] == 1)
					{
						// get its active segment
						int activeSegment = cellActiveSegment[cellId];
						// if it is a sequence segment
						if(segmentSequenceFlag[activeSegment] == 1)
						{
							// set the state of the column to predicted
							buPredicted = 1;
							// set the cell to active state
							cellActivity[cellId] = 1;	
						}
					}
				}

				__syncthreads();

				// if the activity was not predicted
				if(buPredicted == 0)
				{
					// activate all the cells in the column
					for(int c = 0; c < cellsPerColumn; c++)
					{
						cellId = columnId * cellsPerColumn + c;
						cellActivity[cellId] = 1;
					}
				}
			}
		}
	}



	__global__ void PhaseOneKernel(
		// columns
		int *columnActivity,
		int *columnBestMatchingCell,
		// cells
		int *cellActivity,
		int *cellPrediction,
		int *cellLearn,
		int *cellActiveSegment,
		int *cellLearnSegment,
		int *cellBestMatchingSegment,
		// segment
		int *segmentSequenceFlag,
		int *segmentChangeSequenceFlag,
		int *segmentUpdateFlag,
		// constants
		int cellsPerColumn,
		int columns
		)
	{
		int columnId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
	
		if(columnId < columns)
		{
			// for all active columns
			if(columnActivity[columnId] == 1)
			{
				int cellId;
				// bottom up prediction
				int buPredicted = 0;
				// learn cell chosen
				int lcChosen = 0;

				// find predicted cells
				for(int c = 0; c < cellsPerColumn; c++)
				{
					cellId = columnId * cellsPerColumn + c;
					// if the cell is in the predictive state
					if(cellPrediction[cellId] == 1)
					{
						// get its active segment
						int activeSegment = cellActiveSegment[cellId];
						// if it is a sequence segment
						if(segmentSequenceFlag[activeSegment] == 1)
						{
							// set the state of the column to predicted
							buPredicted = 1;
							// set the cell to active state
							cellActivity[cellId] = 1;	


							// IT IS DONE BUT STILL NOT SURE OF THE REASEN FOR THIS
							int learnSegmentId = cellLearnSegment[cellId];
							//if segmentActive(s, t-1, learnState) then
							if(learnSegmentId == activeSegment)
							{
								// lcChosen = true
								cellLearn[cellId] = 1;
								// learnState(c,i,t) = 1
								lcChosen = 1;
							}
						}
					}
				}

				__syncthreads();

				// if the activity was not predicted
				if(buPredicted == 0)
				{
					// activate all the cells in the column
					for(int c = 0; c < cellsPerColumn; c++)
					{
						cellId = columnId * cellsPerColumn + c;
						cellActivity[cellId] = 1;
					}
				}

				__syncthreads();

				// if lcChosen == false then
				if(lcChosen == 0)
				{
					// TO DO: DON'T FORGET TO CHECK IT IT IS NOT EQUAL TO -1
					// i,s = getBestMatchingCell(c,t-1)
					int bestCellId = columnBestMatchingCell[columnId];
					if(bestCellId != -1)
					{
						int bestSegmentId = cellBestMatchingSegment[bestCellId];
						// learnState(c,i,t) = 1
						cellLearn[bestCellId] = 1;
						
						// sUpdate = getSegmentActiveSynapses(c,i,s,t-1,true)
						// sUpdate.sequenceSegment = true
						// segmentUpdateList.add(sUpdate)
						segmentUpdateFlag[bestSegmentId] = 1;
						segmentChangeSequenceFlag[bestSegmentId] = 1;
					}
				}
				
			}
		}
	}



	// !!!! CHANGE THIS KERNEL ACCORDING TO getSegmentActiveSynapses() in the white paper
	__global__ void MarkSynapseChangeOneKernel(
		
		// segment
		int *segmentUpdateFlag,

		// connections
		int *distalMatchingSynapses,
		int *distalSynapseChangeMark,

		// constants
		int connectionsPerSegment,
		int connections
		)
	{
		int connectionId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(connectionId < connections)
		{
			int segmentId = connectionId / connectionsPerSegment;
			if(segmentUpdateFlag[segmentId] == 1)
			{
				distalSynapseChangeMark[connectionId] = distalMatchingSynapses[connectionId];
			}
		}
	}


	__global__ void MarkSynapseChangeOneSegmentKernel(
		
		// segment
		int *segmentUpdateFlag,

		// connections
		int *distalMatchingSynapses,
		int *distalSynapseMatchingMark,

		// constants
		float initPermValue,
		int connectionsPerSegment,
		int segments
		)
	{
		int segmentId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(segmentId < segments)
		{
			if(segmentUpdateFlag[segmentId] == 1)
			{
				int connectionId;
				int matchingSys;
				int newSysFlag = 0;
				for(int c = 0; c < connectionsPerSegment; c++)
				{
					connectionId = segmentId * connectionsPerSegment + c;
					distalSynapseMatchingMark[connectionId] = distalMatchingSynapses[connectionId];
				}
			}
		}
	}

	__global__ void GetDerivedPredictionKernel(
		
		// cells
		int *cellActivity,
		int *cellDerivedPrediction,

		// connections
		int *distalSynapses,
		float *distalPermanence,

		// constants
		int minOverlap,
		float connectedPermanence,
		int connectionsPerSegment,
		int segmentsPerCell,
		int segments
		)
	{
		int segmentId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(segmentId < segments)
		{
			int overlap = 0;
			int connectionId;
			int fromCellId;
			int condition;

			int cellId = segmentId / segmentsPerCell;
			for(int c = 0; c < connectionsPerSegment; c++)
			{
				connectionId = segmentId * connectionsPerSegment + c;
				fromCellId = distalSynapses[connectionId];
				condition = (cellActivity[fromCellId] == 1) * (distalPermanence[connectionId] > connectedPermanence);
				overlap += condition;
			}
			if(overlap >= minOverlap)
			{
				cellDerivedPrediction[cellId] = 1;
			}
		}
	}

	__global__ void SegmentOverlapKernel(
		// cells
		int *cellActivity,
		// segments
		int *segmentActiveOverlap,
		// connections
		int *distalSynapses,
		int *distalActiveSynapses,
		float *distalPermanence,
		// constants
		int minOverlap,
		float connectedPermanence,
		int connectionsPerSegment,
		int segments
		)
	{
		int segmentId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(segmentId < segments)
		{
			int overlap = 0;
			int connectionId;
			int fromCellId;
			int condition;

			for(int c = 0; c < connectionsPerSegment; c++)
			{
				connectionId = segmentId * connectionsPerSegment + c;
				fromCellId = distalSynapses[connectionId];
				condition = (cellActivity[fromCellId] == 1) * (distalPermanence[connectionId] > connectedPermanence);
				//overlap += (cellActivity[fromCellId] == 1) * (distalPermanence[connectionId] > connectedPermanence);
				overlap += condition;
				//distalActiveSynapses[connectionId] = (cellActivity[fromCellId] == 1) * (distalPermanence[connectionId] > connectedPermanence);
				distalActiveSynapses[connectionId] = condition;
				
			}
			/*
			if(overlap < minOverlap)
			{
				overlap = 0;
			}
			*/
			segmentActiveOverlap[segmentId] = overlap * (overlap >= minOverlap);
			//segmentActiveOverlap[segmentId] = overlap;
		}
	}


	__global__ void GetActiveSegmentKernel(
		
		// cells
		int *cellActiveSegment,

		// segments
		int *segmentActiveOverlap,
		int *segmentSequenceFlag,

		// constants
		int connectionsPerSegment,
		int segmentsPerCell,
		int cells
		)
	{
		int cellId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(cellId < cells)
		{
			int maxOverlap = 0;
			int activeSegmentId;
			int biasedOverlap;
			int segmentId;
			int actualOverlap;

			activeSegmentId = -1;
			for(int s = 0; s < segmentsPerCell; s++)
			{
				segmentId = cellId * segmentsPerCell + s;
				actualOverlap = segmentActiveOverlap[segmentId];

				biasedOverlap = actualOverlap + (actualOverlap > 0) * segmentSequenceFlag[segmentId] * connectionsPerSegment;
				if(biasedOverlap > maxOverlap)
				{
					maxOverlap = biasedOverlap;
					activeSegmentId = segmentId;
				}
			}
			cellActiveSegment[cellId] = activeSegmentId;
		}
	}

	
	__global__ void GetLearnSegmentKernel(
		
		// cells
		int *cellLearnSegment,

		// segments
		int *segmentLearnCount,

		// constants
		int segmentsPerCell,
		int cells
		)
	{
		int cellId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(cellId < cells)
		{
			int maxLearnCount = 0;
			int learnSegmentId;
			int segmentId;
			int actualLearnCount;

			learnSegmentId = -1;
			for(int s = 0; s < segmentsPerCell; s++)
			{
				segmentId = cellId * segmentsPerCell + s;
				actualLearnCount = segmentLearnCount[segmentId];

				if(actualLearnCount > maxLearnCount)
				{
					maxLearnCount = actualLearnCount;
					learnSegmentId = segmentId;
				}
			}
			cellLearnSegment[cellId] = learnSegmentId;


		}
	}

	__global__ void SegmentLearnCountKernel(
		
		// cells
		int *cellLearn,

		// segments
		int *segmentLearnCount,

		// connections
		int *distalSynapses,
		int *distalPermanence,

		// constants
		int minOverlap,
		float connectedPermanence,
		int connectionsPerSegment,
		int segments
		)
	{
		int segmentId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(segmentId < segments)
		{
			int learnOverlap = 0;
			int connectionId;
			int fromCellId;

			for(int c = 0; c < connectionsPerSegment; c++)
			{
				connectionId = segmentId * connectionsPerSegment + c;
				fromCellId = distalSynapses[connectionId];
				learnOverlap += (cellLearn[fromCellId] == 1) * (distalPermanence[connectionId] > connectedPermanence);
			}
			/*
			if(learnOverlap < minOverlap)
			{
				learnOverlap = 0;
			}
			*/
			//segmentLearnCount[segmentId] = learnOverlap;
			segmentLearnCount[segmentId] = learnOverlap * (learnOverlap >= minOverlap);
		}
	}


	
	__global__ void GetPredictionKernel(
		// cell
		int *cellPrediction,
		int *cellAtiveSegment,
		// segment
		int *segmentUpdateFlag,
		//constants
		int cells
		)
	{
		int cellId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
		
		if(cellId < cells)
		{
			int activeSegmentId = cellAtiveSegment[cellId];
			if(activeSegmentId != -1)
			{
				cellPrediction[cellId] = 1;
				segmentUpdateFlag[activeSegmentId] = 2;
			}
		}
	}

	

	__global__ void MarkSynapseChangeTwoKernel(
		
		// segments
		int *segmentUpdateFlag,

		// connections
		int *distalActiveSynapses,
		int *distalSynapseChangeMark,

		// constants
		int connectionsPerSegment,
		int connections
		)
	{
		int connectionId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(connectionId < connections)
		{
			int segmentId = connectionId / connectionsPerSegment;
			if(segmentUpdateFlag[segmentId] == 2)
			{
				distalSynapseChangeMark[connectionId] = distalActiveSynapses[connectionId];
			}
		}
	}


	__global__ void GetPredictionSegmentKernel(
		// cell
		int *cellAtiveSegment,
		int *cellPreviousBestMatchingSegment,
		// segment
		int *segmentUpdateFlag,
		//constants
		int cells
		)
	{
		int cellId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
		
		if(cellId < cells)
		{
			int activeSegmentId = cellAtiveSegment[cellId];
			if(activeSegmentId != -1)
			{
				int bestMatchingSegmentId = cellPreviousBestMatchingSegment[cellId];
				if(bestMatchingSegmentId != -1)
				{
					segmentUpdateFlag[bestMatchingSegmentId] = 3;
				}
			}
		}
	}

	__global__ void MarkSynapseChangeThreeKernel(
		
		// segments
		int *segmentUpdateFlag,

		// connections
		int *distalPreviousMatchingSynapses,
		int *distalSynapseChangeMark,

		// constants
		int connectionsPerSegment,
		int connections
		)
	{
		int connectionId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
	
		if(connectionId < connections)
		{
			int segmentId = connectionId / connectionsPerSegment;
			if(segmentUpdateFlag[segmentId] == 3)
			{
				distalSynapseChangeMark[connectionId] = distalPreviousMatchingSynapses[connectionId];
			}
		}
	}


	__global__ void SegmentMatchingKernel(
		//cell
		int *cellActivity,
		//segment
		int *segmentMatchingCount,
		//connection
		int *distalSynapses,
		int *distalMatchingSynapses,
		//constants
		int minMatching,
		int connectionsPerSegment,
		int segments
		)
	{
		int segmentId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;


		if(segmentId < segments)
		{
			int overlap = 0;
			int connectionId;
			int fromCellId;
			for(int c = 0; c < connectionsPerSegment; c++)
			{
				connectionId = segmentId * connectionsPerSegment + c;
				fromCellId = distalSynapses[connectionId];
				if(cellActivity[fromCellId] == 1)
				{
					overlap++;
					distalMatchingSynapses[connectionId] = 1;
				}
			}
			if(overlap < minMatching)
			{
				overlap = 0;
			}
			segmentMatchingCount[segmentId] = overlap;
		}

		/*
		if(segmentId < segments)
		{
			int connectionId;
			int fromCellId;

			int matchingCount = 0;
			for(int c = 0; c < connectionsPerSegment; c++)
			{
				connectionId = segmentId * connectionsPerSegment + c;
				fromCellId = distalSynapses[connectionId];
				if(cellActivity[fromCellId] == 1)
				{
					matchingCount++;
					distalMatchingSynapses[connectionId] = 1;
				}
			}
			if(matchingCount < minMatching)
			{
				matchingCount = 0;
			}
			segmentMatchingCount[segmentId] = matchingCount;
		}
		*/
	}


	__global__ void GetBestMatchingSegmentKernel(
		// cells
		int *cellBestMatchingSegment,
		// segment
		int *segmentMatchingCount,
		// constants
		int segmentsPerCell,
		int cells
		)
	{
		int cellId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(cellId < cells)
		{
			int bestSegmentId = -1;
			int bestMatchingCount = 0;
			int segmentId;
			int actualMatching;

			for(int s = 0; s < segmentsPerCell; s++)
			{
				segmentId = cellId * segmentsPerCell + s;
				actualMatching = segmentMatchingCount[segmentId];
				if(actualMatching > bestMatchingCount)
				{
					bestMatchingCount = actualMatching;
					bestSegmentId = segmentId;
				}
			}
			cellBestMatchingSegment[cellId] = bestSegmentId;
		}
	}

	__global__ void GetBestMatchingCellKernel(
		// column
		int *columnBestMatchingCell,
		// cells
		int *cellBestMatchingSegment,
		// segment
		int *segmentMatchingCount,
		// constants
		int cellsPerColumn,
		int columns
		)
	{
		int columnId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(columnId < columns)
		{
			int bestCellId = -1;
			int bestMatch = 0;
			int cellId;
			int actualMatchSegmentId;
			int actualMatch;

			for(int c = 0; c < cellsPerColumn; c++)
			{
				cellId = columnId * cellsPerColumn + c;
				actualMatchSegmentId = cellBestMatchingSegment[cellId];
				if(actualMatchSegmentId != -1)
				{
					actualMatch = segmentMatchingCount[actualMatchSegmentId];
					if(actualMatch > bestMatch)
					{
						bestCellId = cellId;
						bestMatch = actualMatch;
					}
				}
			}
			columnBestMatchingCell[columnId] = bestCellId;
		}
	}

	__global__ void CopyKernel(
		int *from,
		int *to,
		int count
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < count)
		{
			to[threadId] = from[threadId];
		}
	}
	
	


	__global__ void PhaseThreeKernel(
		// cells
		int *cellPrediction,
		int *cellPreviousPrediction,
		int *cellLearn,
		// segments
		int *segmentSequenceFlag,
		int *segmentChangeSequenceFlag,
		int *segmentUpdateFlag,
		// connections
		float *distalPermanence,
		int *distalSynapseChangeMark,
		// constants
		float permInc,
		float permDec,
		int connectionsPerSegment,
		int segmentsPerCell,
		int segments
		)
	{
		int segmentId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		// for each segment of the cell
		if(segmentId < segments)
		{
			int cellId = segmentId / segmentsPerCell;
			
			// if learnState(c,i,t) == 1 then
			if(cellLearn[cellId] == 1)
			{
				if(segmentUpdateFlag[segmentId] > 0)
				{
					int connectionId;
					// adaptSegments(segmentUpdateList(c,i),true)
					for(int c = 0; c < connectionsPerSegment; c++)
					{
						connectionId = segmentId * connectionsPerSegment + c;
						distalPermanence[connectionId] += permInc * (float)distalSynapseChangeMark[connectionId];
						//distalPermanence[connectionId] += permInc * (float)distalSynapseChangeMark[connectionId] - permDec * (float)(distalSynapseChangeMark[connectionId] == 0);
					}
					segmentSequenceFlag[segmentId] = segmentChangeSequenceFlag[segmentId];
					segmentUpdateFlag[segmentId] = -1 * segmentUpdateFlag[segmentId];
				}
			}
			// else if predictiveState(c,i,t) == 0 and predictiveState(c,i,t-1) == 1 then
			else if(cellPrediction[cellId] == 0 && cellPrediction[cellId] == 1)
			{
				if(segmentUpdateFlag[segmentId] > 0)
				{
					int connectionId;
					// adaptSegments(segmentUpdateList(c,i),false)
					for(int c = 0; c < connectionsPerSegment; c++)
					{
						connectionId = segmentId * connectionsPerSegment + c;
						distalPermanence[connectionId] -= permDec * (float)distalSynapseChangeMark[connectionId];
					}
					segmentUpdateFlag[segmentId] = 0;
					// segmentUpdateList(c,i).delete()
				}
			}
		}
	}


	__global__ void GetPreviousActiveCellIdsKernel(
		
		// cells
		int *cellPreviousActivity,
		int *cellPreviousActiveId,
		// counters
		int *cellPreviousActiveCount,

		// constants
		int cells
		)
	{
		int cellId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(cellId < cells)
		{
			// if the cell was active in previous step
			if(cellPreviousActivity[cellId] == 1)
			{
				int toWritePosition = atomicAdd(&cellPreviousActiveCount[0], 1);
				cellPreviousActiveId[toWritePosition] = cellId;
			}
		}
	}

	__global__ void AddNewConnectionsKernel(
		
		// celss
		int *cellPreviousActiveId,

		// segments
		int *segmentUpdateFlag,

		// connections
		int *distalSynapses,
		float *distalPermanence,
		int *distalSynapseChangeMark,
		float *randomNumber,

		// fields
		int *cellPreviousActiveCount,
		float newPermValue,

		// constants
		int connectionsPerSegment,
		int segmentsPerCell,
		int segments
		)
	{
		int segmentId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(segmentId < segments)
		{
			if(segmentUpdateFlag[segmentId] == -1)
			{
				int activeCount = cellPreviousActiveCount[0];
				int cellId = segmentId / segmentsPerCell;
				int connected = 0;

				int connectionId;
				for(int c = 0; c < connectionsPerSegment; c++)
				{
					// get the global connection id
					connectionId = segmentId * connectionsPerSegment + c;
					// if the synapse was inactive
					if(distalSynapseChangeMark[connectionId] == 0 && connected == 0)
					{
						int connectToId;
						int randomId;
						// get a random synapse from the active ones
						randomId = (int)(randomNumber[connectionId] * (float)activeCount);
						connectToId = cellPreviousActiveId[randomId];

						// if the random cell is different than this one
						if(connectToId != cellId)
						{
							int alreadyConnected = 0;
							int connId;
							// check if the cell is not among the already connected
							for(int conn = 0; conn < connectionsPerSegment; conn++)
							{
								connId = segmentId * connectionsPerSegment + conn;
								if(distalSynapses[connId] == connectToId)
								{
									alreadyConnected == 1;
								}
							}
							// if the cell is not already connected
							if(alreadyConnected == 0)
							{
								distalSynapses[connectionId] = connectToId;
								distalPermanence[connectionId] = newPermValue;
								connected = 1;
							}
						}
					}
				}
			}
		}
	}



	__global__ void AddNewSynapsesKernel(
		
		// cells
		int *cellPreviousActivity,
		int *cellLearn,

		// segments
		int *segmentUpdateFlag,

		// connections
		int *distalSynapses,
		float *distalPermanence,
		int *distalSynapseChangeMark,
		float *randomNumber,

		// constants
		float newPermValue,
		int connectionsPerSegment,
		int segmentsPerCell,
		int cells,
		int segments
		)
	{
		int segmentId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(segmentId < segments)
		{
			if(segmentUpdateFlag[segmentId] == -1)
			{
				int cellId = segmentId / segmentsPerCell;
				int connectionId;
				int connected = 0;
				// go through all the synapses of the segment
				for(int c = 0; c < connectionsPerSegment; c++)
				{
					// get the global connection id
					connectionId = segmentId * connectionsPerSegment + c;
					// if the synapse was inactive
					if(distalSynapseChangeMark[connectionId] == 0 && connected == 0)
					{
						int connectToId;
						// find a random new cell to which the cell want to connect
						connectToId = (int)(randomNumber[connectionId] * (float)cells);
							
						// if the potential connected cell was active in previous step
						// &&
						// if the new cell is not equal than the connected one
						if(cellPreviousActivity[connectToId] == 1 && connectToId != cellId)
						{
							int alreadyConnected = 0;
							int connId;
							// check if the cell is not among the already connected
							for(int conn = 0; conn < connectionsPerSegment; conn++)
							{
								connId = segmentId * connectionsPerSegment + conn;
								if(distalSynapses[connId] == connectToId)
								{
									alreadyConnected == 1;
								}
							}
							// if the cell is not already connected
							if(alreadyConnected == 0)
							{
								distalSynapses[connectionId] = connectToId;
								distalPermanence[connectionId] = newPermValue;
								connected = 1;
							}
						}
					}
				}
			}
		}
	}

	__global__ void SaturatePermanenceKernel(
		
		// connections
		float *distalPermanence,

		// constants
		int connections
		)
	{
		int connectionId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(connectionId < connections)
		{
			distalPermanence[connectionId] = fminf(fmaxf(distalPermanence[connectionId], 0.00f), 1.00f);

		}
	}

	__global__ void FindPredictedColumnsKernel(
		
		// columns
		int *predictionColumnActivity,
		// cells
		int *cellPrediction,
		// constants
		int cellsPerColumn,
		int cells
		)
	{
		int cellId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(cellId < cells)
		{
			if(cellPrediction[cellId] == 1)
			{
				int columnId = cellId / cellsPerColumn;
				predictionColumnActivity[columnId] = 1;
			}
		}
	}

	__global__ void DrawConnectedKernel(
		
		int *columnCanvas,

		// connections
		int *distalSynapses,

		// constants
		int cellsPerColumn,
		int connecionsPerSegment,
		int segmentsPerColumn,
		int columnNr
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < segmentsPerColumn)
		{
			int segmentId = columnNr * segmentsPerColumn + threadId;
			int connectionId;
			int fromCellId;
			int fromColumnId;

			for(int c = 0; c < connecionsPerSegment; c++)
			{
				connectionId = segmentId * connecionsPerSegment + c;
				fromCellId = distalSynapses[connectionId];

				fromColumnId = fromCellId / cellsPerColumn;
				columnCanvas[fromColumnId] = 1;
			}
		}
	}

	__global__ void DifferenceKernel(
		// fields
		int *fieldOne,
		int *fieldTwo,
		int *result,
		// constants
		int count
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;

		if(threadId < count)
		{
			if(fieldOne[threadId] == 1 || fieldTwo[threadId] == 1)
			{
				result[threadId] = (int)(fieldOne[threadId] != fieldTwo[threadId]);	
			}
		}
	
	}

	__global__ void DetectChangeKernel(
		//fields
		int *fieldOne,
		int *fieldTwo,
		int *resultFlag,

		//constants
		int count
		)
	{
		int threadId = blockDim.x*blockIdx.y*gridDim.x	//rows preceeding current row in grid
				+ blockDim.x*blockIdx.x				//blocks preceeding current block
				+ threadIdx.x;
		
		if(threadId < count)
		{
			if(fieldOne[threadId] != fieldTwo[threadId])
			{
				resultFlag[0] = 1;
			}
		}
	}


}
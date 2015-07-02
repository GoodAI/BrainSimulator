#define _SIZE_T_DEFINED 

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>

#include "SegmentDefs.cu"
#include "VisionMath.cu"


extern "C" 
{



	//index for enforce connectivity
	const int dx4[4] = {-1,  0,  1,  0};
	const int dy4[4] = { 0, -1,  0,  1};



	//--- now kernels that do the job...
	__global__ void kInitClusterCenters( float4* floatBuffer, int nWidth, int nHeight, SLICClusterCenter* vSLICCenterList )
	{
		int blockWidth=nWidth/blockDim.x;
		int blockHeight=nHeight/gridDim.x;

		int clusterIdx=blockIdx.x*blockDim.x+threadIdx.x;
		int offsetBlock = blockIdx.x * blockHeight * nWidth + threadIdx.x * blockWidth;
		
		float2 avXY;

		avXY.x=threadIdx.x*blockWidth + (float)blockWidth/2.0;
		avXY.y=blockIdx.x*blockHeight + (float)blockHeight/2.0;

		//use a single point to init center
		int offset=offsetBlock + blockHeight/2 * nWidth+ blockWidth/2 ;

		float4 fPixel=floatBuffer[offset];

		vSLICCenterList[clusterIdx].lab=fPixel;
		vSLICCenterList[clusterIdx].xy=avXY;
		vSLICCenterList[clusterIdx].nPoints=0;
	}





	__global__ void kIterateKmeans( int* maskBuffer, float4* floatBuffer, 
		int nWidth, int nHeight, int nSegs, int nClusterIdxStride,
		SLICClusterCenter* vSLICCenterList, int listSize,
		bool bLabelImg, float weight)
	{


		//for reading cluster centers
		__shared__ float4 fShareLab[3][3];
		__shared__ float2 fShareXY[3][3];

		//pixel index
		__shared__ SLICClusterCenter pixelUpdateList[MAX_BLOCK_SIZE];
		__shared__ float2 pixelUpdateIdx[MAX_BLOCK_SIZE];

		int clusterIdx=blockIdx.y;
		int blockCol=clusterIdx%nClusterIdxStride;
		int blockRow=clusterIdx/nClusterIdxStride;
		//int upperBlockHeight=blockDim.y*gridDim.x;

		int lowerBlockHeight=blockDim.y;
		int blockWidth=blockDim.x;
		int upperBlockHeight=blockWidth;

		int innerBlockHeightIdx=lowerBlockHeight*blockIdx.x+threadIdx.y;

		float M=weight;
		float invWeight=1/((blockWidth/M)*(blockWidth/M));

		int offsetBlock = (blockRow*upperBlockHeight+blockIdx.x*lowerBlockHeight)*nWidth+blockCol*blockWidth;
		int offset=offsetBlock+threadIdx.x+threadIdx.y*nWidth;

		int rBegin=(blockRow>0)?0:1;
		int rEnd=(blockRow+1>(gridDim.y/nClusterIdxStride-1))?1:2;
		int cBegin=(blockCol>0)?0:1;
		int cEnd=(blockCol+1>(nClusterIdxStride-1))?1:2;

		if (threadIdx.x<3 && threadIdx.y<3) {
			if (threadIdx.x>=cBegin && threadIdx.x<=cEnd && threadIdx.y>=rBegin && threadIdx.y<=rEnd) {
				int cmprIdx=(blockRow+threadIdx.y-1)*nClusterIdxStride+(blockCol+threadIdx.x-1);

				fShareLab[threadIdx.y][threadIdx.x]=vSLICCenterList[cmprIdx].lab;
				fShareXY[threadIdx.y][threadIdx.x]=vSLICCenterList[cmprIdx].xy;
			}
		}

		__syncthreads();

		if (innerBlockHeightIdx>=blockWidth)
			return;

		if (offset>=nWidth*nHeight)
			return;

		// finding the nearest center for current pixel
		float fY=blockRow*upperBlockHeight+blockIdx.x*lowerBlockHeight+threadIdx.y;
		float fX=blockCol*blockWidth+threadIdx.x;

		if (fY<nHeight && fX<nWidth)
		{
			float4 fPoint=floatBuffer[offset];
			float minDis=9999;
			int nearestCenter=-1;
			int nearestR, nearestC;

			for (int r=rBegin;r<=rEnd;r++)
			{
				for (int c=cBegin;c<=cEnd;c++)
				{
					int cmprIdx=(blockRow+r-1)*nClusterIdxStride+(blockCol+c-1);

					//compute SLIC distance
					float fDab=(fPoint.x-fShareLab[r][c].x)*(fPoint.x-fShareLab[r][c].x)
						+(fPoint.y-fShareLab[r][c].y)*(fPoint.y-fShareLab[r][c].y)
						+(fPoint.z-fShareLab[r][c].z)*(fPoint.z-fShareLab[r][c].z);
					//fDab=sqrt(fDab);

					float fDxy=(fX-fShareXY[r][c].x)*(fX-fShareXY[r][c].x)
						+(fY-fShareXY[r][c].y)*(fY-fShareXY[r][c].y);
					//fDxy=sqrt(fDxy);

					float fDis=fDab+invWeight*fDxy;

					if (fDis<minDis)
					{
						minDis=fDis;
						nearestCenter=cmprIdx;
						nearestR=r;
						nearestC=c;
					}
				}
			}

			if (nearestCenter>-1) {
				int pixelIdx=threadIdx.y*blockWidth+threadIdx.x;

				if(pixelIdx < MAX_BLOCK_SIZE) {
					pixelUpdateList[pixelIdx].lab=fPoint;
					pixelUpdateList[pixelIdx].xy.x=fX;
					pixelUpdateList[pixelIdx].xy.y=fY;

					pixelUpdateIdx[pixelIdx].x=nearestC;
					pixelUpdateIdx[pixelIdx].y=nearestR;
				}

				if (bLabelImg)
					maskBuffer[offset]=nearestCenter;
			}
		}
		else {
			int pixelIdx=threadIdx.y*blockWidth+threadIdx.x;

			if(pixelIdx < MAX_BLOCK_SIZE) {
				pixelUpdateIdx[pixelIdx].x=-1;
				pixelUpdateIdx[pixelIdx].y=-1;
			}
		}

		__syncthreads();
	}






	__global__ void kUpdateClusterCenters( float4* floatBuffer,int* maskBuffer, int nWidth, int nHeight, int nSegs, SLICClusterCenter* vSLICCenterList, int listSize)
	{

		int blockWidth=nWidth/blockDim.x;
		int blockHeight=nHeight/gridDim.x;

		int clusterIdx=blockIdx.x*blockDim.x+threadIdx.x;

		int offsetBlock = threadIdx.x * blockWidth+ blockIdx.x * blockHeight * nWidth;

		float2 crntXY=vSLICCenterList[clusterIdx].xy;
		float4 avLab;
		float2 avXY;
		int nPoints=0;

		avLab.x=0;
		avLab.y=0;
		avLab.z=0;

		avXY.x=0;
		avXY.y=0;

		int yBegin=0 < (crntXY.y - blockHeight) ? (crntXY.y - blockHeight) : 0;
		int yEnd= nHeight > (crntXY.y + blockHeight) ? (crntXY.y + blockHeight) : (nHeight-1);	
		int xBegin=0 < (crntXY.x - blockWidth) ? (crntXY.x - blockWidth) : 0;
		int xEnd= nWidth > (crntXY.x + blockWidth) ? (crntXY.x + blockWidth) : (nWidth-1);

		//update to cluster centers
		for (int i = yBegin; i < yEnd ; i++)
		{
			for (int j = xBegin; j < xEnd; j++)
			{
				int offset=j + i * nWidth;

				float4 fPixel=floatBuffer[offset];
				int pIdx=maskBuffer[offset];

				if (pIdx==clusterIdx)
				{
					avLab.x+=fPixel.x;
					avLab.y+=fPixel.y;
					avLab.z+=fPixel.z;

					avXY.x+=j;
					avXY.y+=i;

					nPoints++;
				}
			}
		}

		if(nPoints == 0)
			return;

		avLab.x/=nPoints;
		avLab.y/=nPoints;
		avLab.z/=nPoints;

		avXY.x/=nPoints;
		avXY.y/=nPoints;

		vSLICCenterList[clusterIdx].lab=avLab;
		vSLICCenterList[clusterIdx].xy=avXY;
		vSLICCenterList[clusterIdx].nPoints=nPoints;
	}















	//=======================================================
	//  create descriptors
	///------------------------------------------------------
	/// Add how edges around look like...
	__device__ void Desc_get_hists_stats_for_each_segment (float2 xy, int id, float* edge_im , float* feat_desc, int dim_desc, int width){
		int dim_id_start = 5;
		int ngh_max = 5;

		feat_desc[id*dim_desc+dim_id_start+0]     = edge_im[((int)xy.x) + ((int)xy.y)*width];
		feat_desc[id*dim_desc+dim_id_start+1]     = abs(feat_desc[id*dim_desc+dim_id_start+0]);
		feat_desc[id*dim_desc+dim_id_start+2]     = edge_im[((int)xy.x)+ngh_max + ((int)xy.y+ngh_max)*width];
		feat_desc[id*dim_desc+dim_id_start+3]     = abs(feat_desc[id*dim_desc+dim_id_start+2]);
		feat_desc[id*dim_desc+dim_id_start+4]     = edge_im[((int)xy.x)+ngh_max + ((int)xy.y-ngh_max)*width];
		feat_desc[id*dim_desc+dim_id_start+5]     = abs(feat_desc[id*dim_desc+dim_id_start+4]);
		feat_desc[id*dim_desc+dim_id_start+6]     = edge_im[((int)xy.x)-ngh_max + ((int)xy.y+ngh_max)*width];
		feat_desc[id*dim_desc+dim_id_start+7]     = abs(feat_desc[id*dim_desc+dim_id_start+6]);
		feat_desc[id*dim_desc+dim_id_start+8]     = edge_im[((int)xy.x)-ngh_max + ((int)xy.y-ngh_max)*width];
		feat_desc[id*dim_desc+dim_id_start+9]     = abs(feat_desc[id*dim_desc+dim_id_start+8]);
}

	//          m_kernel_desc.Run(devSLICCCenter.DevicePointer, Owner.features_xy, Owner.features_desc , Owner.nSegs , 2 , Owner.dim_feat_desc); /// fill image with average color
	__global__ void Desc (SLICClusterCenter* vSLICCenterList , float* feat_xy , float* feat_desc , int size , int dim_xy , int dim_desc , int width , int height){//, float* edge_im){
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;
		if (id<size){
			float2 xy  = vSLICCenterList[id].xy;
			float4 lab = vSLICCenterList[id].lab;

			feat_xy[id*dim_xy+0] = xy.x;// (xy.x-width/2) / (width/2);
			feat_xy[id*dim_xy+1] = xy.y;// (xy.y-height/2) / (height/2);
			feat_xy[id*dim_xy+2] = (int)vSLICCenterList[id].nPoints;//  0.2;// id;

			//--- old desc
			float desc_prev[3];
			desc_prev[0] = feat_desc[id*dim_desc+0];
			desc_prev[1] = feat_desc[id*dim_desc+1];
			desc_prev[2] = feat_desc[id*dim_desc+2];
			
			feat_desc[id*dim_desc+0] = lab.x/1;
			feat_desc[id*dim_desc+1] = lab.y/1;
			feat_desc[id*dim_desc+2] = lab.z/1;

			feat_desc[id*dim_desc+3] = Dist_between_two_vec(&desc_prev[0] , feat_desc+id*dim_desc , 3);
		}

	}




	__global__ void Copy_intMat2Float(float* A , int* B , int size){
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;

		if (id < size) {			
			A[id] = (float)B[id];
		}
	}











	//=======================================================
	//  From image to float4 buffer
	///------------------------------------------------------
		__global__ void kBw2XYZ(float* im , float4* outputImg , int size)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;

		if (id < size) {
		
			float _b= im[id];
			float _g= im[id];
			float _r= im[id];

			float x=_r*0.412453	+_g*0.357580	+_b*0.180423;
			float y=_r*0.212671	+_g*0.715160	+_b*0.072169;
			float z=_r*0.019334	+_g*0.119193	+_b*0.950227;

			float4 fPixel;
			fPixel.x=x;
			fPixel.y=y;
			fPixel.z=z;

			outputImg[id]=fPixel;
		}
	}
	__global__ void kRgb2XYZ(float* imR , float* imG , float* imB , float4* outputImg , int size)
	{
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;

		if (id < size) {			

			float _b= imB[id];
			float _g= imG[id];
			float _r= imR[id];

			float x=_r*0.412453	+_g*0.357580	+_b*0.180423;
			float y=_r*0.212671	+_g*0.715160	+_b*0.072169;
			float z=_r*0.019334	+_g*0.119193	+_b*0.950227;

			float4 fPixel;
			fPixel.x=x;
			fPixel.y=y;
			fPixel.z=z;

			outputImg[id]=fPixel;
		}		
	}
	__global__ void kRgb2LAB(float* imR , float* imG , float* imB , float4* outputImg , int size)
	{
			
		int id = blockDim.x*blockIdx.y*gridDim.x	+ blockDim.x*blockIdx.x	+ threadIdx.x;

		if (id < size) {			

			float _b= imB[id];
			float _g= imG[id];
			float _r= imR[id];

			float x=_r*0.412453	+_g*0.357580	+_b*0.180423;
			float y=_r*0.212671	+_g*0.715160	+_b*0.072169;
			float z=_r*0.019334	+_g*0.119193	+_b*0.950227;

			float l,a,b;

			x/=0.950456;
			float y3=exp(log(y)/3.0);
			z/=1.088754;

			x = x>0.008856 ? exp(log(x)/3.0) : (7.787*x+0.13793);
			y = y>0.008856 ? y3 : 7.787*y+0.13793;
			z = z>0.008856 ? z/=exp(log(z)/3.0) : (7.787*z+0.13793);

			l = y>0.008856 ? (116.0*y3-16.0) : 903.3*y;
			a=(x-y)*500.0;
			b=(y-z)*200.0;


			float4 fPixel;
			fPixel.x=l;
			fPixel.y=a;
			fPixel.z=b;
		
			outputImg[id]=fPixel;
		}
	}	
}







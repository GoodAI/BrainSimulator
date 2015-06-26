#define _SIZE_T_DEFINED 

#include <cuda.h> 
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_functions.h> 
#include <float.h>



#define OFFSET(x,y,dim) ((x)+(y)*(dim))
#define MAX_BLOCK_SIZE 512   /// Need to correspond to MySeg.cs MAX_BLOCK_SIZE


extern "C" 
{


	//Round a / b to nearest higher integer value
	__host__ int iDivUp(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
	//Round a / b to nearest lower integer value
	__host__ int iDivDown(int a, int b) { return a / b; }
	//Align a to nearest higher multiple of b
	__host__ int iAlignUp(int a, int b) { return (a % b != 0) ?  (a - a % b + b) : a; }
	//Align a to nearest lower multiple of b
	__host__ int iAlignDown(int a, int b)  {return a - a % b; }



	typedef struct
	{
		float4 lab;
		float2 xy;
		int nPoints;
		int x1, y1, x2, y2;
	}SLICClusterCenter;

}
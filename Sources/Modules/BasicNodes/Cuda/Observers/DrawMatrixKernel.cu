#define _SIZE_T_DEFINED 
#ifndef __CUDACC__ 
#define __CUDACC__ 
#endif 
#ifndef __cplusplus 
#define __cplusplus 
#endif

#include <cuda.h> 
#include <math_constants.h> 
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h> 
#include <texture_fetch_functions.h> 
#include <builtin_types.h> 
#include <vector_types.h> 
#include <vector_functions.h> 

#define CHARACTER_MAP_DIGITS_OFFSET (16)
#define CHARACTER_MAP_X (56)
#define CHARACTER_MAP_SPACE (0)
#define CHARACTER_MAP_PLUS (11)
#define CHARACTER_MAP_MINUS (13)
#define CHARACTER_MAP_DOT (14)
#define CHARACTER_MAP_a (65)
#define CHARACTER_MAP_E (37)
#define CHARACTER_MAP_f (70)
#define CHARACTER_MAP_I (41)
#define CHARACTER_MAP_N (46)
#define CHARACTER_MAP_n (78)


#define DATA_TYPE_FLOAT 0
#define DATA_TYPE_INTEGER 1

extern "C"  
{
	__constant__ unsigned int* D_CANVAS;
	__constant__ unsigned int* D_CHARACTER_MAP;
	__constant__ int D_CHARACTER_MAP_NB_CHARS;
	__constant__ int* D_VALUES_INTEGER;
	__constant__ float* D_VALUES_FLOAT;
	__constant__ int D_NB_VALUES;
	__constant__ int D_START_COL;
	__constant__ int D_LENGTH_COL;
	__constant__ int D_START_ROW;
	__constant__ int D_LENGTH_ROW;
	__constant__ int D_TEXTURE_WIDTH;
	__constant__ int D_TEXTURE_HEIGHT;
	__constant__ int D_CHARACTER_WIDTH;
	__constant__ int D_CHARACTER_HEIGHT;
	__constant__ int D_CHARACTER_SIZE;
	__constant__ int D_MATRIX_COLS;
	__constant__ int D_MATRIX_ROWS;
	__constant__ int D_NB_CHARACTER_PER_BOX;
	__constant__ int D_NB_DECIMALS;

	
	__device__ float roundToNDecimals(float number, int nbDecimals);
	__device__ void regularValue(int characterId, float value, int& characterMapAddress);
	__device__ void regularValueNormalDisplay(int characterId, float value, int& characterMapAddress);
	__device__ void regularValueScientificDisplay(int characterId, float value, int& characterMapAddress);
	__device__ void nanValue(int characterId, int& characterMapAddress, unsigned int& rgbColor);
	__device__ void infinityValue(int characterId, float value, int& characterMapAddress, unsigned int& rgbColor);



	//kernel code
	__global__ void DrawMatrixKernel(int dataType)
	{
		
		int pixelId = blockDim.x*blockIdx.y*gridDim.x	
				+ blockDim.x*blockIdx.x				
				+ threadIdx.x;

		if (pixelId >= D_TEXTURE_WIDTH * D_TEXTURE_HEIGHT)
			return;

		int boxId = pixelId / (D_NB_CHARACTER_PER_BOX * D_CHARACTER_SIZE);

		if (boxId >= D_NB_VALUES) // In case where we dont have a perfect rectangle, some boxes must be left empty
		{
			// TODO Write spaces
			return;
		}
		
		int col = D_START_COL + boxId % D_LENGTH_COL;
		int row = D_START_ROW + boxId / D_LENGTH_COL;

		int valueIndex = row * D_MATRIX_COLS + col;

		float value;
		switch (dataType)
		{
		case DATA_TYPE_FLOAT:
			value = D_VALUES_FLOAT[valueIndex];
			break;
		case DATA_TYPE_INTEGER:
			value = D_VALUES_INTEGER[valueIndex];
			break;
		default:
			// Shouldn't happen
			value = 0;
			break;
		}
		

		
		// Find the character id
		//
		// Example: for drawing +1.75e2, we need to draw 5 characters:
		// Syntax  + 1 . 7 5 e 2 _
		// Index   0 1 2 3 4 5 6 7
		
		int inBoxPixelId = pixelId % (D_NB_CHARACTER_PER_BOX * D_CHARACTER_SIZE);
		int characterId = inBoxPixelId / D_CHARACTER_SIZE;
		int inCharacterPixelId = inBoxPixelId % D_CHARACTER_SIZE;

		unsigned int rgbColor = 0x0; // Black;

		// Find the character to draw
		int characterMapAddress;
		
		if (isinf(value))
			infinityValue(characterId, value, characterMapAddress, rgbColor);
		else if (isnan(value))
			nanValue(characterId, characterMapAddress, rgbColor);
		else
			regularValue(characterId, value, characterMapAddress);

		if (characterMapAddress < 0)
			return;

		int inCharacterPixelPositionX = inCharacterPixelId % D_CHARACTER_WIDTH;
		int inCharacterPixelPositionY = inCharacterPixelId / D_CHARACTER_WIDTH;

		
		int inDigitMapPositionX = characterMapAddress * D_CHARACTER_WIDTH + inCharacterPixelPositionX;
		int inDigitMapPositionY = inCharacterPixelPositionY;
		
		unsigned int characterMapIndex = inDigitMapPositionX + inDigitMapPositionY * (D_CHARACTER_MAP_NB_CHARS * D_CHARACTER_WIDTH);
		float factor = D_CHARACTER_MAP[characterMapIndex];

		unsigned int red = (float)(rgbColor >> 16) * factor + 255.0 * (1.0 - factor);
		unsigned int green = (float)((rgbColor & 0x00FF00) >> 8) * factor + 255.0 * (1.0 - factor);
		unsigned int blue = (float)(rgbColor & 0x0000FF) * factor + 255.0 * (1.0 - factor);

		unsigned int pixelColor = 0xFF000000 + (red << 16) + (green << 8) + (blue);
		
		

		int canvasPositionX = ((boxId % D_LENGTH_COL) * (D_NB_CHARACTER_PER_BOX + 1) + characterId) * D_CHARACTER_WIDTH + inCharacterPixelPositionX;
		int canvasPositionY = (boxId / D_LENGTH_COL) * D_CHARACTER_HEIGHT + inCharacterPixelPositionY;
		

		D_CANVAS[canvasPositionX + canvasPositionY * D_TEXTURE_WIDTH] = pixelColor;
	}


	
	__device__ float roundToNDecimals(float number, int nbDecimals)
	{
		float scale = pow((double)10, (double)nbDecimals);
		return round(number * scale) / scale;
	}

	__device__ void regularValue(int characterId, float value, int& characterMapAddress)
	{
		if (value != 0)
			value = roundToNDecimals(value, D_NB_DECIMALS);

		// 0 is a special value
		if (abs(value) < pow((double)10, (double)(-D_NB_DECIMALS)))
		{
			if (characterId == 1)
				characterMapAddress = CHARACTER_MAP_DIGITS_OFFSET + 0; // 0
			else if (D_NB_DECIMALS > 0)
			{
				if (characterId == 2)
					characterMapAddress = CHARACTER_MAP_DOT;
				else if (characterId > 0 && characterId <= 2 + D_NB_DECIMALS)
					characterMapAddress = CHARACTER_MAP_DIGITS_OFFSET + 0; // 0
				else
					characterMapAddress = CHARACTER_MAP_SPACE;
			}
			else
			{
				characterMapAddress = CHARACTER_MAP_SPACE;
			}
		}
		else
		{
			// Different of 0
			if (characterId == 0) // Sign
			{
				if (value >= 0)
					characterMapAddress = CHARACTER_MAP_SPACE;
				else
					characterMapAddress = CHARACTER_MAP_MINUS;
			}
			else
			{
				if (value < 0)
					value = -value;
			
			
				if (1 + floor(log10(value)) + ((D_NB_DECIMALS > 0) ? (D_NB_DECIMALS + 1) : 0) < D_NB_CHARACTER_PER_BOX)
				{
					// Enough space to write the plain value
					regularValueNormalDisplay(characterId, value, characterMapAddress);
				}
				else
				{
					// Not enough space, we write it in scientific mode
					regularValueScientificDisplay(characterId, value, characterMapAddress);
				}
			}
		}
	}

	__device__ void regularValueNormalDisplay(int characterId, float value, int& characterMapAddress)
	{
		int scale = floor(log10(value));
		if (scale >= 0)
		{
			if (characterId <= scale + 1)
			{
				// Integer part
				int digit = (int)floor((value / pow((double)10, scale - characterId + 1))) % 10;
				characterMapAddress = CHARACTER_MAP_DIGITS_OFFSET + digit; // Digit address
			}
			else if (characterId > scale + 2)
			{
				// Decimal part
				int decimalNum = characterId - scale - 2;

				if (decimalNum <= D_NB_DECIMALS)
				{
					int digit = (int)floor(value * pow((double)10, decimalNum)) % 10;
					characterMapAddress = CHARACTER_MAP_DIGITS_OFFSET + digit; // Digit address
				}
				else
				{
					characterMapAddress = CHARACTER_MAP_SPACE;
				}
			}
			else
			{
				// Dot
				if (D_NB_DECIMALS > 0)
					characterMapAddress = CHARACTER_MAP_DOT;
				else
					characterMapAddress = CHARACTER_MAP_SPACE;
			}
		}
		else
		{
			if (characterId == 1)
				characterMapAddress = CHARACTER_MAP_DIGITS_OFFSET + 0; // 0
			else if (characterId == 2)
				characterMapAddress = CHARACTER_MAP_DOT;
			else if (characterId - 2 <= D_NB_DECIMALS)
			{
				// Decimal part
				int decimalNum = characterId - 2;

				if (decimalNum <= D_NB_DECIMALS)
				{
					int digit = (int)floor(value * pow((double)10, decimalNum)) % 10;
					characterMapAddress = CHARACTER_MAP_DIGITS_OFFSET + digit; // Digit address
				}
				else
				{
					characterMapAddress = CHARACTER_MAP_SPACE;
				}
			}
			else
				characterMapAddress = CHARACTER_MAP_SPACE;

		}
	}


	__device__ void regularValueScientificDisplay(int characterId, float value, int& characterMapAddress)
	{
		if (D_NB_DECIMALS > 0 && characterId == 2) // Dot
		{
			characterMapAddress = CHARACTER_MAP_DOT;
		}
		else if (characterId == D_NB_CHARACTER_PER_BOX - 4) // E
		{
			characterMapAddress = CHARACTER_MAP_E;
		}
		else
		{
			int power;
			if (value != 0)
				power = floor(log10(value));
			else
				power = 0;


			if (characterId == D_NB_CHARACTER_PER_BOX - 2) // 1st power digit
			{
				if (power < 0)
					power = -power;
					characterMapAddress = CHARACTER_MAP_DIGITS_OFFSET + power / 10; // Digit address
			}
			else if (characterId == D_NB_CHARACTER_PER_BOX - 1) // 2nd power digit
			{
				if (power < 0)
					power = -power;
					characterMapAddress = CHARACTER_MAP_DIGITS_OFFSET + power % 10; // Digit address
			}
			else if (characterId == D_NB_CHARACTER_PER_BOX - 3) // power sign
			{
				if (power >= 0)
					characterMapAddress = CHARACTER_MAP_PLUS;
				else
					characterMapAddress = CHARACTER_MAP_MINUS;
			}
			else
			{
				float number = value / pow((double)10, (double)power);
				if (characterId == 1) // Mantisse integer
				{
					int firstDigit = number;
					characterMapAddress = CHARACTER_MAP_DIGITS_OFFSET + firstDigit; // Digit address
				}
				else // Mantisse decimal
				{
					int digit = (int)(number * pow((double)(10), (double)(characterId - 2))) % 10;
					characterMapAddress = CHARACTER_MAP_DIGITS_OFFSET + digit;
				}

			}
		}
	}

	__device__ void nanValue(int characterId, int& characterMapAddress, unsigned int& rgbColor)
	{
		rgbColor = 0xFFFF0000;

		if (characterId == 0 || characterId == 2)
		{
			characterMapAddress = CHARACTER_MAP_N;
		}
		else if (characterId == 1)
		{
			characterMapAddress = CHARACTER_MAP_a;
		}
		else
		{
			characterMapAddress = CHARACTER_MAP_SPACE;
		}
	}

	
	__device__ void infinityValue(int characterId, float value, int& characterMapAddress, unsigned int& rgbColor)
	{
		rgbColor = 0xFFFF0000;

		if (characterId == 0) // Sign
		{
			if (value >= 0)
				characterMapAddress = CHARACTER_MAP_PLUS;
			else
				characterMapAddress = CHARACTER_MAP_MINUS;
		}
		else if (characterId == 1)
		{
			characterMapAddress = CHARACTER_MAP_I;
		}
		else if (characterId == 2)
		{
			characterMapAddress = CHARACTER_MAP_n;
		}
		else if (characterId == 3)
		{
			characterMapAddress = CHARACTER_MAP_f;
		}
		else
		{
			characterMapAddress = CHARACTER_MAP_SPACE;
		}
	}
}
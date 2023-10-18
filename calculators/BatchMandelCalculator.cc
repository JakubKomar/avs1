/**
 * @file BatchMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date DATE
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>
#include <immintrin.h>
#include "BatchMandelCalculator.h"

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	data= (int *)_mm_malloc(height * width * sizeof(int), height);
}

BatchMandelCalculator::~BatchMandelCalculator() {
	// @TODO cleanup the memory
	_mm_free(data);
	data = NULL;
}

int * BatchMandelCalculator::calculateMandelbrot () {
	constexpr int32_t BLOCK_L3_SIZE =512;
	constexpr int32_t BLOCK_L2_SIZE =128;
	int *pdata = data;
	const int32_t c_height=height;
	const int32_t c_width=width;
	const int32_t c_limit=limit;
	const float c_x_start = x_start;
	const float c_y_start = y_start;
	const float c_dx = dx;
	const float c_dy = dy;


	for (int32_t i_l3 = 0; i_l3 < (c_height/2)/BLOCK_L3_SIZE; i_l3++)
	{
		for (int32_t j_l3 = 0; j_l3 < c_width/BLOCK_L3_SIZE; j_l3++)
		{
			for (int32_t i_l2 = 0; i_l2 < BLOCK_L3_SIZE/BLOCK_L2_SIZE; i_l2++)
			{
				for (int32_t j_l2 = 0; j_l2 < BLOCK_L3_SIZE/BLOCK_L2_SIZE; j_l2++)
				{
					for (int32_t i_l1 = 0; i_l1 < BLOCK_L2_SIZE; i_l1++)
					{						
						for (int32_t j_l1 = 0; j_l1< BLOCK_L2_SIZE; j_l1++)
						{
							const int32_t j=j_l3*BLOCK_L3_SIZE + j_l2*BLOCK_L2_SIZE + j_l1;
							const int32_t i=i_l3*BLOCK_L3_SIZE + i_l2*BLOCK_L2_SIZE + i_l1;
							//std::cout<<"i:"<<i<<" j:"<<j<<std::endl;
							float x=c_x_start + j * c_dx;
							float y=c_y_start + i * c_dy;
							float zReal = x;
							float zImag = y;

							int32_t l;
							for (l = 0; l < limit; ++l)
							{
								float r2 = zReal * zReal;
								float i2 = zImag * zImag;

								if (r2 + i2 > 4.0f)
									break;

								zImag = 2.0f * zReal * zImag + y;
								zReal = r2 - i2 + x;
							}
							pdata[i * c_width + j ] = l;
						}
					}
				}
			}
		}
	}

	for (int32_t i = 0; i < c_height/2; i++)
	{
		int* srcRowPtr = pdata + i * c_width;  // Ukazatel na zdrojový řádek
        int* destRowPtr = pdata + (c_height - i - 1) * c_width;  // Ukazatel na cílový řádek
		
		#pragma omp simd
		for (int32_t j = 0; j < c_width; j++)
		{
			destRowPtr[j]=srcRowPtr[j];
		}
	}
	return pdata;
}

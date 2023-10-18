/**
 * @file LineMandelCalculator.cc
 * @author FULL NAME <xlogin00@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date DATE
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <immintrin.h>
#include <stdlib.h>


#include "LineMandelCalculator.h"


LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	//data = (int *)(malloc(height * width * sizeof(int)));
	data= (int *)_mm_malloc(height * width * sizeof(int), height);
}

LineMandelCalculator::~LineMandelCalculator() {
	//free(data);
	_mm_free(data);
	data = NULL;
}

int * LineMandelCalculator::calculateMandelbrot () {
	int *pdata = data;
	const int c_height=height;
	const int c_width=width;
	const int c_limit=limit;
	const float c_x_start = x_start;
	const float c_y_start = y_start;
	const float c_dx = dx;
	const float c_dy = dy;


	//#pragma omp simd collapse(3)
	//#pragma omp parallel for simd
	for (int i = 0; i < c_height/2; i++)
	{
		#pragma omp simd
		for (int j = 0; j < c_width; j++)
		{
			float x=c_x_start + j * c_dx;
			float y=c_y_start + i * c_dy;
			float zReal = x;
			float zImag = y;

			int l;
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

	//#pragma omp parallel for
	for (int i = 0; i < c_height/2; i++)
	{
		int* srcRowPtr = pdata + i * c_width;  // Ukazatel na zdrojový řádek
        int* destRowPtr = pdata + (c_height - i - 1) * c_width;  // Ukazatel na cílový řádek
		
		#pragma omp simd
		for (int j = 0; j < c_width; j++)
		{
			destRowPtr[j]=srcRowPtr[j];
		}
	}
	return pdata;
}

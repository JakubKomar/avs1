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

#include <stdlib.h>
#include <stdexcept>
#include <immintrin.h>

#include "LineMandelCalculator.h"


LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	//data = (int *)(malloc(height * width * sizeof(int)));
	data=  new int32_t[height*width];
	//(int *)_mm_malloc(height * width * sizeof(int), 64 * sizeof(int));

	x =new _Float32[width];
	zReal =new _Float32[width];
	zImag =new _Float32[width];
	results =new int32_t[width];
}

LineMandelCalculator::~LineMandelCalculator() {
	delete data;
	data = NULL;
	delete x;
	delete zReal;
	delete zImag;
	delete results;
}

int * LineMandelCalculator::calculateMandelbrot () {
	int *pdata = data;
	const int32_t c_height=height;
	const int32_t c_width=width;
	const int32_t c_limit=limit;
	const _Float32 c_x_start = x_start;
	const _Float32 c_y_start = y_start;
	const _Float32 c_dx = dx;
	const _Float32 c_dy = dy;


	_Float32 * x =new _Float32[c_width];
	_Float32 * zReal =new _Float32[c_width];
	_Float32 * y =new _Float32[c_width];
	_Float32 * zImag =new _Float32[c_width];
	int32_t * results =new int32_t[c_width];

	for (int32_t i = 0; i < c_height/2; i++){
		const _Float32 y=c_y_start + i * c_dy;

		#pragma omp simd
		for (int32_t j = 0; j < c_width; j++){
			x[j]=c_x_start + j * c_dx;
			zReal[j]=x[j];
			zImag[j] = y;
			results[j] =limit;
		}

		int32_t done=0;
		for (int32_t l = 0; l < limit; ++l){
			
			#pragma omp simd
			for (int32_t j = 0; j < c_width; j++){
				_Float32 r2 = zReal[j] * zReal[j];
				_Float32 i2 = zImag[j] * zImag[j];

				if (r2 + i2 > 4.0f && results[j]==limit){
					results[j]=l;
					done++;
				}
					

				zImag[j] = 2.0f * zReal[j] * zImag[j] + y;
				zReal[j] = r2 - i2 + x[j];
			}
			if(done>=c_width){break;}
		}	
		
		const int32_t arrayShift=i * c_width;
		#pragma omp simd
		for (int32_t j = 0; j < c_width; j++){
			pdata[arrayShift + j ] = results[j];
		}	
	}

	for (int32_t i = 0; i < c_height/2; i++){
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

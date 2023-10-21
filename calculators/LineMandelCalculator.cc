/**
 * @file LineMandelCalculator.cc
 * @author Jakub Komárek <xkomar33@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 21.10.2023
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>
#include <immintrin.h>

#include "LineMandelCalculator.h"

#define ALIGEN_SIZE 32
LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	data	=(int *)	_mm_malloc(height * width * sizeof(int), ALIGEN_SIZE * sizeof(int));
	results =(int32_t *)_mm_malloc(width * sizeof(int32_t), ALIGEN_SIZE * sizeof(int32_t));

	x  		=(_Float32 *)_mm_malloc(width * sizeof(_Float32), ALIGEN_SIZE * sizeof(_Float32));
	zReal  	=(_Float32 *)_mm_malloc(width * sizeof(_Float32), ALIGEN_SIZE * sizeof(_Float32));
	zImag  	=(_Float32 *)_mm_malloc(width * sizeof(_Float32), ALIGEN_SIZE * sizeof(_Float32));
}

LineMandelCalculator::~LineMandelCalculator() {
	_mm_free(data);
	_mm_free(results);
	_mm_free(x);
	_mm_free(zReal);
	_mm_free(zImag);
	data = NULL;
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


	for (int32_t i = 0; i < c_height/2; i++){
		const _Float32 y=c_y_start + i * c_dy;

		#pragma vector aligned
		for (int32_t j = 0; j < c_width; j++){
			zImag[j] = y;
		}

		#pragma vector aligned
		for (int32_t j = 0; j < c_width; j++){
			results[j] =limit;
		}

		#pragma vector aligned
		for (int32_t j = 0; j < c_width; j++){
			const _Float32 xCalc=c_x_start + j * c_dx;
			x[j]=xCalc;
			zReal[j]=xCalc;
		}

		int32_t done=0;
		for (int32_t l = 0; l < limit; ++l){
			
			#pragma omp simd reduction(+:done)
			for (int32_t j = 0; j < c_width; j++){
				const _Float32 r2 = zReal[j] * zReal[j];
				const _Float32 i2 = zImag[j] * zImag[j];

				if (r2 + i2 > 4.0f && results[j]==limit){
					results[j]=l;
					done++;
				}
					
				zImag[j] = 2.0f * zReal[j] * zImag[j] + y;
				zReal[j] = r2 - i2 + x[j];
			}
			if(done>=c_width){
				break;
			}
		}	
		
		const int32_t arrayShift=i * c_width;
		#pragma vector aligned
		for (int32_t j = 0; j < c_width; j++){
			pdata[arrayShift + j ] = results[j];
		}	
	}

	for (int32_t i = 0; i < c_height/2; i++){
		int* srcRowPtr = pdata + i * c_width;  // Ukazatel na zdrojový řádek
        int* destRowPtr = pdata + (c_height - i - 1) * c_width;  // Ukazatel na cílový řádek
		
		#pragma vector aligned
		for (int j = 0; j < c_width; j++){
			destRowPtr[j]=srcRowPtr[j];
		}
	}
	return pdata;
}

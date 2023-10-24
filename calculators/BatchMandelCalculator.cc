/**
 * @file BatchMandelCalculator.cc
 * @author Jakub Komárek <xkomar33@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 21.10.2023
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>
#include <immintrin.h>
#include "BatchMandelCalculator.h"

#define LINE_GROUP_SIZE 128
#define ALIGEN_SIZE 64
#define SIMD_LEN_SIZE 32
BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	data	=	(int32_t *) _mm_malloc(height  * width * sizeof(int), ALIGEN_SIZE);

	x		=	(_Float32 *)_mm_malloc(LINE_GROUP_SIZE * sizeof(_Float32), ALIGEN_SIZE);
	zReal  	=	(_Float32 *)_mm_malloc(LINE_GROUP_SIZE * sizeof(_Float32), ALIGEN_SIZE);
	zImag  	=	(_Float32 *)_mm_malloc(LINE_GROUP_SIZE * sizeof(_Float32), ALIGEN_SIZE);
}	

BatchMandelCalculator::~BatchMandelCalculator() {
	_mm_free(data);

	_mm_free(x);
	_mm_free(zReal);
	_mm_free(zImag);
	data = NULL;
}

int * BatchMandelCalculator::calculateMandelbrot () {

	const int32_t c_height=height;
	const int32_t c_width=width;
	const int32_t c_limit=limit;

	const _Float32 c_x_start = x_start;
	const _Float32 c_y_start = y_start;
	const _Float32 c_dx = dx;
	const _Float32 c_dy = dy; 

	_Float32 * x_ptr=x;
	_Float32 * zReal_ptr=zReal;
	_Float32 * zImag_ptr=zImag;

	for (int32_t i = 0; i < c_height/2; i++){ // cykly přes všechny řádky
		const _Float32 y=c_y_start + i * c_dy;
		const int32_t rowOffset= i*c_width;

		
		for (int32_t j_L3 = 0; j_L3 < c_width/LINE_GROUP_SIZE; j_L3++){ //cykly přes všechny skupiny na řádku
			const int32_t column_offset= j_L3 * LINE_GROUP_SIZE;
			int32_t * o_data = (int32_t *) (data+rowOffset+column_offset);

			#pragma omp simd aligned(zImag_ptr:ALIGEN_SIZE) simdlen(SIMD_LEN_SIZE) 
			for (int32_t j = 0; j < LINE_GROUP_SIZE; j++){
				zImag_ptr[j] = y;
			}

			#pragma omp simd aligned(x_ptr:ALIGEN_SIZE,zReal_ptr:ALIGEN_SIZE) simdlen(SIMD_LEN_SIZE) 
			for (int32_t j = 0; j < LINE_GROUP_SIZE; j++){
				const _Float32 xCalc=c_x_start + (column_offset+j) * c_dx;
				x_ptr[j]=xCalc;
				zReal_ptr[j]=xCalc;
			}
			

			#pragma omp simd aligned(o_data:ALIGEN_SIZE) simdlen(SIMD_LEN_SIZE) 
			for (int32_t j = 0; j < LINE_GROUP_SIZE; j++){
				o_data[j] =limit;
			}
			
			int32_t done=0;
			for (int32_t l = 0; l < limit; ++l){  //cykli v limitu 

				#pragma omp simd reduction(+:done) aligned(zReal_ptr:ALIGEN_SIZE,zImag_ptr:ALIGEN_SIZE,x_ptr:ALIGEN_SIZE,o_data:ALIGEN_SIZE) simdlen(SIMD_LEN_SIZE) 
				for (int32_t j = 0; j < LINE_GROUP_SIZE; j++){ //pro každou položku v podskupině

					const _Float32 r2 = zReal_ptr[j] * zReal_ptr[j];
					const _Float32 i2 = zImag_ptr[j] * zImag_ptr[j];

					if (r2 + i2 > 4.0f && o_data[j]==limit){
						o_data[j]=l;
						done++;
					}

					zImag_ptr[j] = 2.0f * zReal_ptr[j] * zImag_ptr[j] + y;
					zReal_ptr[j] = r2 - i2 + x_ptr[j];
				}	
				if(done>=LINE_GROUP_SIZE){
					break;
				}
			}	
		}
	}

	for (int32_t i = 0; i < c_height/2; i++){
		const int* srcRowPtr = data + i * c_width;  // Ukazatel na zdrojový řádek
        int* destRowPtr = data + (c_height - i - 1) * c_width;  // Ukazatel na cílový řádek

		#pragma omp simd aligned(destRowPtr:ALIGEN_SIZE,srcRowPtr: ALIGEN_SIZE ) simdlen(SIMD_LEN_SIZE) 
		for (int j = 0; j < c_width; j++){
			destRowPtr[j]=srcRowPtr[j];
		}
	}
	return data;
}

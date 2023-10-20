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

#define L2_SIZE 128
#define L3_SIZE 512
BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	//data = (int *)(malloc(height * width * sizeof(int)));
	//data = (int *)(malloc(height * width * sizeof(int)));
	data=  new int32_t[height*width];
	//(int *)_mm_malloc(height * width * sizeof(int), 64 * sizeof(int));

	x =new _Float32[L3_SIZE];
	zReal =new _Float32[L3_SIZE];
	zImag =new _Float32[L3_SIZE];
	results =new int32_t[L3_SIZE];
}

BatchMandelCalculator::~BatchMandelCalculator() {
	delete data;
	data = NULL;
	delete x;
	delete zReal;
	delete zImag;
	delete results;
}

int * BatchMandelCalculator::calculateMandelbrot () {

	int *pdata = data;
	const int32_t c_height=height;
	const int32_t c_width=width;
	const int32_t c_limit=limit;
	const _Float32 c_x_start = x_start;
	const _Float32 c_y_start = y_start;
	const _Float32 c_dx = dx;
	const _Float32 c_dy = dy;

	for (int32_t i = 0; i < c_height/2; i++){ // cykly přes všechny řádky
		const _Float32 y=c_y_start + i * c_dy;
		const int32_t rowOffset= i*c_width;

		for (int32_t j_L3 = 0; j_L3 < c_width/L3_SIZE; j_L3++){ //cykly přes všechny skupiny na řádku
			const int32_t j_l3_offset= j_L3 * L3_SIZE;
			
			#pragma omp simd
			for (int32_t j = 0; j < L3_SIZE; j++){
				zImag[j] = y;
			}

			#pragma omp simd
			for (int32_t j = 0; j < L3_SIZE; j++){
				results[j] =limit;
			}

			#pragma omp simd
			for (int32_t j = 0; j < L3_SIZE; j++){
				const _Float32 xCalc=c_x_start + j * c_dx;
				x[j]=xCalc;
				zReal[j]=xCalc;
			}

			int32_t done=0;
			for (int32_t l = 0; l < limit; ++l){  //cykli v limitu 

				#pragma omp scimd
				for (int32_t j = 0; j < L3_SIZE; j++){ //pro každou položku v podskupině 			

					_Float32 r2 = zReal[j] * zReal[j];
					_Float32 i2 = zImag[j] * zImag[j];

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

			#pragma omp simd
			for (int32_t j = 0; j < L3_SIZE; j++){
				pdata[ rowOffset + j_l3_offset + j] = results[j];
			}		
		}
	}

	for (int32_t i = 0; i < c_height/2; i++){
		int* srcRowPtr = pdata + i * c_width;  // Ukazatel na zdrojový řádek
        int* destRowPtr = pdata + (c_height - i - 1) * c_width;  // Ukazatel na cílový řádek
		
		#pragma omp simd
		for (int j = 0; j < c_width; j++){
			destRowPtr[j]=srcRowPtr[j];
		}
	}
	
	return pdata;
}

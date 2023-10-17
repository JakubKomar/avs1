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


#include "LineMandelCalculator.h"


LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	data = (int *)(malloc(height * width * sizeof(int)));
	//data= (int *)_mm_malloc(height * width * sizeof(int), height);
}

LineMandelCalculator::~LineMandelCalculator() {
	free(data);
	//_mm_free(data);
	data = NULL;
}

template <typename T>
static inline int mandelbrot(const T real,const T imag,const int limit)
{
	T zReal = real;
	T zImag = imag;

	int i;
	for (i = 0; i < limit; ++i)
	{
		T r2 = zReal * zReal;
		T i2 = zImag * zImag;

		if (r2 + i2 > 4.0f)
			break;

		zImag = 2.0f * zReal * zImag + imag;
		zReal = r2 - i2 + real;
	}
	return i;
}//dwad

int * LineMandelCalculator::calculateMandelbrot () {
	int *pdata = data;
	const int c_height=height;
	const int c_width=width;
	const int c_limit=limit;
	const float c_x_start = x_start;
	const float c_y_start = y_start;
	const float c_dx = dx;
	const float c_dy = dy;


	for (int i = 0; i < c_height; i++)
	{
		#pragma omp simd
		for (int j = 0; j < c_width; j+=4)
		{
			float x[4];
			float y[4];
			int value[4];

			// Compute four real and imaginary values
			#pragma unroll
			for (int k = 0; k < 4; k++)
			{
				x[k] = c_x_start + (j + k) * c_dx;
				y[k] = c_y_start + i * c_dy;
			}

			// Compute Mandelbrot values for the four points
			#pragma unroll
			for (int k = 0; k < 4; k++)
			{
				value[k] = mandelbrot(x[k], y[k], c_limit);
			}

			// Store the Mandelbrot values
			#pragma unroll
			for (int k = 0; k < 4; k++)
			{
				pdata[i * c_width + j + k] = value[k];
			}
		}
	}
	return data;
}

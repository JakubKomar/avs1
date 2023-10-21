/**
 * @file LineMandelCalculator.h
 * @author Jakub Komárek <xkomar33@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 21.10.2023
 */

#include <BaseMandelCalculator.h>

class LineMandelCalculator : public BaseMandelCalculator
{
public:
    LineMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator();
    int *calculateMandelbrot();

private:
    int *data;
    
	float * x;
	float * zReal;
	float * zImag;
	int32_t * results;
};
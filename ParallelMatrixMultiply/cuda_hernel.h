#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

void MatrixMultiplyWithCuda(unsigned long long *c, const unsigned long long *a, const unsigned long long *b, unsigned int m, unsigned int n, unsigned int o);
void MatrixRandomFill(unsigned long long *matrix, unsigned int m, unsigned int n);
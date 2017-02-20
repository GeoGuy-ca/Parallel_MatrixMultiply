
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <algorithm>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <Windows.h>

__global__ void multiplyKernel(unsigned long long *c, const unsigned long long *a, const unsigned long long *b, unsigned int m, unsigned int n, unsigned int o)
{
	int me = (blockIdx.y * blockDim.y) + (blockIdx.x * blockDim.x) + threadIdx.x;
	int col = me / m;
	int row = me % m;
	for (int element = 0; element < n; element++) {
		c[me] += a[element*m + row] * b[col*n + element];
	}
}

void MatrixMultiplyWithCuda(unsigned long long *c, const unsigned long long *a, const unsigned long long *b, unsigned int m, unsigned int n, unsigned int o)
{
	unsigned long long *dev_a = 0;
	unsigned long long *dev_b = 0;
	unsigned long long *dev_c = 0;
	cudaSetDevice(0);
	cudaMalloc((void**)&dev_c, m * o * sizeof(long long));
	cudaMalloc((void**)&dev_a, m * n * sizeof(long long));
	cudaMalloc((void**)&dev_b, n * o * sizeof(long long));

	cudaMemcpy(dev_a, a, m * n * sizeof(long long), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, n * o * sizeof(long long), cudaMemcpyHostToDevice);

	dim3 threads(1024);//max 1024 threads per block!
	dim3 blocks(m / threads.x + 1, o / threads.x + 1);
	multiplyKernel<<<blocks, threads>>>(dev_c, dev_a, dev_b, m, n, o);
	cudaDeviceSynchronize();

	cudaMemcpy(c, dev_c, m * o * sizeof(long long), cudaMemcpyDeviceToHost);

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);
}


void MatrixRandomFill(unsigned long long *matrix, unsigned int m, unsigned int n)
{
	unsigned long long *dev_matrix = 0;
	cudaSetDevice(0);
	cudaMalloc((void**)&dev_matrix, m * n * sizeof(long long));

	curandGenerator_t gen;
	curandCreateGenerator(&gen, CURAND_RNG_QUASI_SCRAMBLED_SOBOL64);
	curandSetGeneratorOffset(gen, GetTickCount64());
	curandSetQuasiRandomGeneratorDimensions(gen, m * n);
	curandGenerateLongLong(gen, dev_matrix, m * n);
	
	cudaDeviceSynchronize();
	cudaMemcpy(matrix, dev_matrix, m * n * sizeof(long long), cudaMemcpyDeviceToHost);

	cudaFree(dev_matrix);
}
#include <omp.h>

void MatrixMultiplyOpenMPColumn(unsigned long long *c, const unsigned long long *a, const unsigned long long *b, unsigned int m, unsigned int n, unsigned int o, int threads) {
	int row, col, element;
	omp_set_dynamic(0);
#pragma omp parallel for private(row, element) num_threads(threads)
	for (col = 0; col < m; col++) {
		for (row = 0; row < o; row++) {
			for (element = 0; element < n; element++) {
				c[col + row*m] += a[element*m + col] * b[row*n + element];
			}
		}
	}
}


void MatrixMultiplyOpenMPElement(unsigned long long *c, const unsigned long long *a, const unsigned long long *b, unsigned int m, unsigned int n, unsigned int o, int threads) {
	int size = m*o;
	int element, col, row, partsum;
	omp_set_dynamic(0);
#pragma omp parallel for private(row, col, partsum) num_threads(threads)
	for (int element = 0; element < size; element++) {
		col = element / m;
		row = element % m;
		for (partsum = 0; partsum < n; partsum++) {
			c[element] += a[partsum*m + row] * b[col*n + partsum];
		}
		
	}
		
}
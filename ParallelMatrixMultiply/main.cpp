#include "stdafx.h"
#include "cuda_hernel.h"
#include <time.h>
#include <Windows.h>
#include "mpi.h"
#include "OpenMP_Matrix.h"
#include <fstream>
#include <iostream>

//For Testing
void printMatrix(unsigned long long *c, int m, int n) {
	printf("\n");
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			printf("%d,", c[m*j + i]);
		}
		printf("\n");
	}
	printf("\n");
}

int test(unsigned int m, unsigned int n, unsigned int o, std::ofstream& output, int threads) {
	//Ints to track timings
	int begin_process; 
	int end_process;
	
	unsigned long long *a = new unsigned long long[m * n]; //Allocate array A to heap
	unsigned long long *b = new unsigned long long[n * o]; //Allocate array A to heap
	MatrixRandomFill(a, m, n); //Place random Data in Array A
	MatrixRandomFill(b, n, o); //Place random Data in Array B
	unsigned long long *openmp_columns = new unsigned long long[m*o];
	memset(openmp_columns, 0, m * o * sizeof(long long));
	output << "OpenMP-column, ";
	output << threads;
	output << ", ";
	output << m;
	output << ", ";
	output << n;
	output << ", ";
	output << o;
	begin_process = GetTickCount();
	MatrixMultiplyOpenMPColumn(openmp_columns, a, b, m, n, o, threads);
	end_process = GetTickCount();
	output << ",";
	output << end_process - begin_process;
	output << "\n";
	free(openmp_columns);

	unsigned long long *openmp_elements = new unsigned long long[m*o];
	memset(openmp_elements, 0, m * o * sizeof(long long));
	output << "OpenMP-Element, ";
	output << threads;
	output << ", ";
	output << m;
	output << ", ";
	output << n;
	output << ", ";
	output << o;
	begin_process = GetTickCount();
	MatrixMultiplyOpenMPElement(openmp_elements, a, b, m, n, o, threads);
	end_process = GetTickCount();
	output << ",";
	output << end_process - begin_process;
	output << "\n";
	free(openmp_elements);

	if (threads == 1) {
		unsigned long long *cuda = new unsigned long long[m * o];
		memset(cuda, 0, m * o * sizeof(long long));
		output << "cuda, 1920, ";
		output << m;
		output << ", ";
		output << n;
		output << ", ";
		output << o;
		begin_process = GetTickCount();

		MatrixMultiplyWithCuda(cuda, a, b, m, n, o);
		end_process = GetTickCount();
		output << ",";
		output << end_process - begin_process;
		output << "\n";
		free(cuda);
	}
	free(a);
	free(b);
	
	
	return 0;
}

int main()
{
	int threads[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 125, 256 };
	std::ofstream file;
	file.open("resultsCUDA.csv");
	file << "Process, Threads, M, N, O, Time(ms)\n";
	int MAX_DIMENSION = 14;
	for (int t = 0; t < sizeof(threads)/sizeof(threads[0]); t++) {
		printf("Threads: %d\n- - - - - - - - - - - - - - - - - - - - - -\n", threads[t]);
		for (unsigned int m = MAX_DIMENSION; m >= 7; m--) {
			printf("%d:", m);	
					for (unsigned int i = 0; i < 3; i++) {
						printf(".");
						test(pow(2, m), pow(2, m), pow(2, m), file, threads[t]);
					}
			printf("\n");
		}
		printf("- - - - - - - - - - - - - - - - - - - - - -\n\n\n");
	}
	file << "End of test\n";
	file.close();
}




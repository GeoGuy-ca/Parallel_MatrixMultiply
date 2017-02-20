#include "stdafx.h"
#include "cuda_hernel.h"
#include <time.h>
#include <Windows.h>
#include "mpi.h"
#include "OpenMP_Matrix.h"
#include <fstream>
#include <iostream>

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
	output << "MPI_COLUMN, ";
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
	output << "MPI_ELEMENT, ";
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
		output << "CUDA, 1920, ";
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
	std::ofstream file;
	file.open("results10.csv");
	file << "Process, Threads, M, N, O, Time(ms)\n";
	int MAX_DIMENSION = 10;
	for (int t = 0; t >= 0; t--) {
		int tmp = pow(2, t);
		printf("Threads: %d\n- - - - - - - - - - - - - - - - - - - - - -\n", tmp);
		for (unsigned int m = MAX_DIMENSION; m >= 1; m--) {
			printf("%d:", m);
			for (unsigned int n = m; n >= 1; n--) {
				printf(" %d", n);
				for (unsigned int o = n; o >= 1; o--) {
					printf(".");
					for (unsigned int i = 0; i < 3; i++) {
						test(pow(2, m), pow(2, m), pow(2, m), file, pow(2, t));
					}
				}
			}
			printf("\n");
		}
		printf("- - - - - - - - - - - - - - - - - - - - - -\n\n\n");
	}
	file << "End of test\n";
	file.close();

	std::ofstream file2;
	file2.open("results11.csv");
	file2 << "Process, Threads, M, N, O, Time(ms)\n";
	MAX_DIMENSION = 11;
	for (int t = 10; t >= 0; t--) {
		int tmp = pow(2, t);
		printf("Threads: %d\n- - - - - - - - - - - - - - - - - - - - - -\n", tmp);
		for (unsigned int m = MAX_DIMENSION; m >= 1; m--) {
			printf("%d:", m);
			for (unsigned int n = m; n >= 1; n--) {
				printf(" %d", n);
				for (unsigned int o = n; o >= 1; o--) {
					printf(".");
					for (unsigned int i = 0; i < 3; i++) {
						test(pow(2, m), pow(2, m), pow(2, m), file2, pow(2, t));
					}
				}
			}
			printf("\n");
		}
		printf("- - - - - - - - - - - - - - - - - - - - - -\n\n\n");
	}
	file2 << "End of test\n";
	file2.close();

	std::ofstream file3;
	file3.open("results12.csv");
	file3 << "Process, Threads, M, N, O, Time(ms)\n";
	MAX_DIMENSION = 12;
	for (int t = 10; t >= 0; t--) {
		int tmp = pow(2, t);
		printf("Threads: %d\n- - - - - - - - - - - - - - - - - - - - - -\n", tmp);
		for (unsigned int m = MAX_DIMENSION; m >= 1; m--) {
			printf("%d:", m);
			for (unsigned int n = m; n >= 1; n--) {
				printf(" %d", n);
				for (unsigned int o = n; o >= 1; o--) {
					printf(".");
					for (unsigned int i = 0; i < 3; i++) {
						test(pow(2, m), pow(2, m), pow(2, m), file3, pow(2, t));
					}
				}
			}
			printf("\n");
		}
		printf("- - - - - - - - - - - - - - - - - - - - - -\n\n\n");
	}
	file3 << "End of test\n";
	file3.close();
}




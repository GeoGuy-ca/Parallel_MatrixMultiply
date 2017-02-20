#pragma once
void MatrixMultiplyOpenMPColumn(unsigned long long *c, const unsigned long long *a, const unsigned long long *b, unsigned int m, unsigned int n, unsigned int o, int threads);

void MatrixMultiplyOpenMPElement(unsigned long long *c, const unsigned long long *a, const unsigned long long *b, unsigned int m, unsigned int n, unsigned int o, int threads);
#ifndef XNOR_GEMM_H
#define XNOR_GEMM_H

void gemm_bin2(int M, int N, int K,
               float *A, int lda,
               float *B, int ldb,
               float *C, int ldc);


void xnor_gemm_impl(unsigned int* A, unsigned int* B, float* C, int m, int n, int k, int count);

void xnor_gemm_concat(unsigned int* AT, float* B, float* C, int m, int n, int k);

void concatenate_rows_kernel_cpu(float *a, unsigned int *b, int rows, int cols);

void concatenate_cols_kernel_cpu(float *a, unsigned int *b, int rows, int cols);

#endif //

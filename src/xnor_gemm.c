/*
 * xnor_gemm.c
 *
 *  Created on: Apr 6, 2016
 *      Author: busta
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "xnor_gemm.h"
#include "utils.h"

#include <nmmintrin.h>

// 32 single float array ->  32 bits unsigned int
unsigned int concatenate_cpu(float* array, int count)
{
    unsigned int rvalue=0;
    unsigned int sign;

    for (int i = 0; i < count; i++)
    {
        sign = (unsigned int) (array[i]>0);
        rvalue = rvalue | (sign<<i);
    }
    return rvalue;
}

void concatenate_cols_kernel_cpu(float *a, unsigned int *b, int rows, int cols)
{
	int rows_m = rows / 32;
    int rows_cut = rows_m * 32;
    int rest = rows % 32;
    #pragma omp parallel for
	for( int j = 0; j < cols; j++) {
        int row_m = 0;
        for (int i = 0; i < rows_cut; i += 32, row_m += 1) {
            float array[32];
            for (int k = 0; k < 32; k++)
                array[k] = a[j + cols * (i + k)];

            b[row_m * cols + j] = concatenate_cpu(array, 32);
        }
        if (rest > 0){
            float array[32];
            for (int k = rest; k < 32; k++)
                array[k] = 0;
            for (int k = 0; k < rest; k++)
                array[k] = a[j + cols * (rows_m * 32 + k)];
            b[rows_m * cols + j] = concatenate_cpu(array, rest);
        }
	}
}


void concatenate_rows_kernel_cpu(float *a, unsigned int *b, int rows, int cols)
{
	int cols_m = cols / 32;
    int rest = cols % 32;
    int cols_count = cols_m;
    if(rest > 0)
        cols_count += 1;

    #pragma omp parallel for
    for( int i = 0; i < rows; i++)
    {
    	unsigned int *br = &b[i * cols_count];
    	float *ar = &a[i * cols];
    	for( int j = 0; j < cols_m; j++)
    	{
    		br[j] = concatenate_cpu(&ar[j*32], 32);
    	}
        if(rest > 0)
            br[cols_m] = concatenate_cpu(&ar[cols_m*32], rest);

    }

}

inline unsigned int bitcount32(unsigned int i) {

    //Parallel binary bit add
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    return (((i + (i >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;

}

void xnor_gemm_impl(unsigned int* A, unsigned int* B, float* C, int m, int n, int k, int count)
{
	for (int celly = 0; celly < m; ++celly)
	{
		for (int cellx = 0; cellx < n; ++cellx)
		{
			register unsigned int Cvalue = 0;
			for (int x = 0; x < k; x++)
			{
				Cvalue += __builtin_popcount(A[celly * k + x]^B[cellx + x * n]);
                //Cvalue += bitcount32(A[celly * k + x]^B[cellx + x * n]);
			}
			// Write Csub to device memory
			// Each thread writes one element
			int cIdx = celly * n + cellx;
            float val = -(2*(float)Cvalue-count);
			C[cIdx] = val;
		}
	}
}

void gemm_bin2(int M, int N, int K,
              float *A, int lda,
              float *B, int ldb,
              float *C, int ldc)
{
    int i,j,k;
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = A[i*lda+k];
            if( A_PART > 0)
            {
                for(j = 0; j < N; ++j) {
                    assert(B[k * ldb + j] <= 1);
                    C[i * ldc + j] += B[k * ldb + j];
                    /*
                    if( B[k * ldb + j] > 0 )
                        C[i * ldc + j] += 1;
                    else
                        C[i * ldc + j] -= 1;
                    */
                }
            }else
            {
                assert(B[k * ldb + j] <= 1);
                C[i * ldc + j] += B[k * ldb + j];
                /*
                for(j = 0; j < N; ++j) {
                    if( B[k * ldb + j] > 0 )
                        C[i * ldc + j] -= 1;
                    else
                        C[i * ldc + j] += 1;
                }*/
            }
        }
    }
}

void test_gemm_conct(int m, int n, int k)
{

    int k_red = k / 32;
    if ( k % 32 > 0 )
        k_red += 1;

    float* A = calloc( m  * k, sizeof(float));
    float* B = calloc( n  * k, sizeof(float));
    printf("A:\n");
    for( int i = 0; i < m; i++ ){
        for(int j = 0; j < k; j++ ){
            float val = rand_uniform(-1, 1);
            A[i * k + j] = val > 0 ? 1 : -1;
        }
    }
    for( int i = 0; i < m; i++ ){
        for(int j = k - 1; j >= 0; j-- ){
            printf("%d", A[i * k + j] == 1 ? 1 : 0);
            if( (j) % 32 == 0 && j > 0)
                printf(" ");
        }
        printf("\n");
    }
    printf("B:\n");
    for(int j = 0; j < k; j++ ){
        for( int i = 0; i < n; i++ ){
            float val = rand_uniform(-1, 1);
            B[j * n + i] = val > 0 ? 1 : -1;
        }
    }
    for( int i = 0; i < n; i++ ){
        for(int j = k - 1; j >= 0; j-- ){
            printf("%d", B[j * n + i] == 1 ? 1 : 0);
            if( (j) % 32 == 0 && j > 0)
                printf(" ");
        }
        printf("\n");
    }


    unsigned int* at = calloc( m  * k_red, sizeof(unsigned int));
    unsigned int* bt = calloc(n * k_red, sizeof(unsigned int));
    float* c_check = calloc( m  * n, sizeof(float));
    float* c_check2 = calloc( m  * n, sizeof(float));

    concatenate_rows_kernel_cpu(A, at, m, k);
    printf("at0: %#010x %u\n", at[0], at[0]);
    printf("at1: %#010x %u\n", at[1], at[1]);
    concatenate_cols_kernel_cpu(B, bt, k, n);
    printf("b0: %#010x %u\n", bt[0], bt[0]);
    printf("b1: %#010x %u\n", bt[1], bt[1]);
    gemm_bin2(m, n, k, A, k, B, n, c_check, n);

    xnor_gemm_impl(at, bt, c_check2, m, n, k_red, k);

    for(int i = 0; i < m; i++ ){
        for(int j = 0; j < n; j++) {
            if (c_check[i * n + j] != c_check2[i * n + j]) {
                printf("%d %d, %f, %f\n", i, j, c_check[i * n + j], c_check2[i * n + j]);
                printf("fail for %d, %d, %d \n", m, n, k);
            }
            assert(c_check[i * n + j] == c_check2[i * n + j]);
        }
    }
}

//#define VERBOSE_T
#define NUMERIC_TEST 1
/**
 * A is shape (m,k), B is shape (k,n) and C is shape (m,n)
 */
void xnor_gemm_concat(unsigned int* AT, float* B, float* C, int m, int n, int k)
{
	int k_red = k / 32;
	if ( k % 32 > 0 )
        k_red += 1;

	unsigned int* bt = calloc(n * k_red, sizeof(unsigned int));

	concatenate_cols_kernel_cpu(B, bt, k, n);
#ifdef VERBOSE_T
    clock_t timeg=clock();
#endif
	xnor_gemm_impl(AT, bt, C, m, n, k_red, k);

#ifdef VERBOSE_T
    printf(" gemmimpl: %lf sec\n", sec(clock()-timeg));
#endif
    free(bt);
}



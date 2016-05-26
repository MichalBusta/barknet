#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "xnor_conv_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
#include "xnor_gemm.h"
#include <assert.h>
}

__global__ void binarize_xnor_input_kernel(float *input, int n, int size, float * mean_input)
{
    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= size) return;
    int i = 0;
    float mean = 0;
    for(i = 0; i < n; ++i){
        mean += fabs(input[i*size + s]);
    }
    mean = mean / n;
    mean_input[s] = mean;
    for(i = 0; i < n; ++i){
        input[i*size + s] = (input[i*size + s] > 0) ? mean : -mean;
    }
}

void binarize_xnor_input_gpu(float *input, int n, int size, float* mean_input)
{
    binarize_xnor_input_kernel<<<cuda_gridsize(size), BLOCK>>>(input, n, size, mean_input);
    check_error(cudaPeekAtLastError());
}


// 32 single float array ->  32 bits unsigned int
__device__ unsigned int concatenate(float* array)
{
    unsigned int rvalue=0;
    unsigned int sign;

    for (int i = 0; i < 32; i++)
    {
        sign = (array[i]>0);
        rvalue = rvalue | (sign<<i);
    }

    return rvalue;
}

__global__ void concatenate_rows_kernel_gpu(float *a, unsigned int *b, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<size) b[i] = concatenate(&a[i*32]);
}

__global__ void concatenate_cols_kernel(float *a, unsigned int *b, int rows, int n)
{

    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(j<n){
        float * array = new float[32];
        for(int i=0; i<rows; i+=32){
            for(int k=0; k<32;k++)
                array[k] = a[j + n*(i+k)];
            b[j+n*i/32]=concatenate(array);
        }
        delete[] array;
    }
}

void concatenate_rows_gpu(float *a, unsigned int *b, int rows, int cols)
{

    int k_red = cols / 32;
    if( cols > k_red * 32 )
        k_red += 1;
    int block = 64, grid = rows * cols / (block * 32)  + 1;
    concatenate_rows_kernel_gpu<<<grid, block>>>(a, b, rows * k_red);
    cudaDeviceSynchronize();
}

// A is shape (m,k_red), B is shape (k_red,n) and C is shape (m,n)
__global__ void xnor_gemm_impl_gpu(unsigned int* A, unsigned int* B, float* C, int m, int n, int k_red, float* a, float *c_norm, int k) {

    int s = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if (s >= m * n) return;

    int celly =  s / n;
    int cellx = s % n;

    unsigned int Cvalue = 0;
    for (int x = 0; x < k_red; ++x) {
        Cvalue += __popc(A[celly * k_red + x]^B[cellx + x * n]);
    }
    float val = -(2*(float)Cvalue-k);
    C[celly * n + cellx] = val * fabs(c_norm[cellx]) * fabs(a[celly*k]);
}


void concatenate_cols_gpu(float *a, unsigned int *b, int rows, int cols)
{
    int block = 64;
    int grid = cols / block + 1;
    concatenate_cols_kernel << < grid, block >> > (a, b, rows, cols);
    cudaDeviceSynchronize();
}

//#define NUMERIC_TEST 1

/**
 * A is shape (m,k), B is shape (k,n) and C is shape (m,n)
 */
void xnor_gemm_concat_gpu(unsigned int* AT, float* B, float* C, int m, int n, int k, float* a, float* c_norm, unsigned int* bt)
{
    int k_red = k / 32;
    if ( k % 32 > 0 )
        k_red += 1;

    concatenate_cols_gpu(B, bt, k, n); //todo(michal) cols can be in transposed order

#ifdef VERBOSE_T
    clock_t timeg=clock();
#endif

    int size = m * n;
    xnor_gemm_impl_gpu<<<cuda_gridsize(size), BLOCK>>>(AT, bt, C, m, n, k_red, a, c_norm, k);
    cudaDeviceSynchronize();

#ifdef NUMERIC_TEST

    float* b_pull = (float *) calloc( k * n, sizeof(float));
    status = cudaMemcpy(b_pull, B, k * n * sizeof(float), cudaMemcpyDeviceToHost);
    check_error(status);

    unsigned  int* b_check = (unsigned  int *) calloc( n * k_red, sizeof(unsigned int));
    concatenate_cols_kernel_cpu(b_pull, b_check,  k, n);

    unsigned  int* b_check_cuda = (unsigned  int *) calloc( n * k_red, sizeof(unsigned int));
    status = cudaMemcpy(b_check_cuda, bt,  n * k_red * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    check_error(status);

    for(int j = 0; j < n * k_red; j++ ){
        if( b_check_cuda[j] != b_check[j]  )
        {
            printf("b_check: %d, %f, %f\n", j, b_check_cuda[j], b_check[j]);
        }
        assert(b_check_cuda[j] == b_check[j]);
    }


    float* c_check_cuda = (float *) calloc( n * m, sizeof(float));
    status = cudaMemcpy(c_check_cuda, C,  n * m * sizeof(float), cudaMemcpyDeviceToHost);
    check_error(status);

    unsigned  int* filters_concat = (unsigned int *) calloc( m * k_red, sizeof(unsigned int *));
    status = cudaMemcpy(filters_concat, AT,  m * k_red * sizeof(float), cudaMemcpyDeviceToHost);
    check_error(status);

    float* c = (float *) calloc( n * m, sizeof(float));
    xnor_gemm_impl(filters_concat, b_check, c, m, n, k_red, k);

    float* c_norm_check =  (float *) calloc( n, sizeof(float));
    status = cudaMemcpy(c_norm_check, c_norm,  n * sizeof(float), cudaMemcpyDeviceToHost);
    check_error(status);

    float* a_check =  (float *) calloc( m * k, sizeof(float));
    status = cudaMemcpy(a_check, a,  m * k * sizeof(float), cudaMemcpyDeviceToHost);
    check_error(status);

    for(int ii = 0; ii < m; ii++){
        register float A_PART = fabs(a_check[ii*k]);
        for(int j = 0; j < n; j++) {
            //c[ii * n + j] *= (fabs(c_norm_check[j]) *  A_PART);
            assert(c[ii * n + j] != c_check_cuda[ii * n + j] );
            assert(!isinf(c[ii * n + j]));
        }
    }

    free(b_pull);
    free(b_check_cuda);
    free(c_check_cuda);
    free(filters_concat);
    free(c);
    free(c_norm_check);

#endif //NUMERIC_TEST

#ifdef VERBOSE_T
    printf(" gemmimpl: %lf sec\n", sec(clock()-timeg));
#endif
}


void forward_xnor_conv_layer_gpu(xnor_conv_layer l, network_state state)
{
    int i;
    int m = l.n;
    int k = l.size*l.size*l.c;
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    int n = out_h * out_w;

    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    binarize_filters_gpu(l.filters_gpu, l.n, l.c*l.size*l.size, l.binary_filters_gpu);
    swap_binary(&l);
    concatenate_rows_gpu(l.filters_gpu, l.filters_concat, l.n, l.size*l.size*l.c);

#ifdef NUMERIC_TEST
    //test if filters are same

    int k_red = k / 32;
    if( k > k_red * 32 )
        k_red += 1;
    unsigned  int* s_concat = (unsigned  int *) calloc( m  * k_red, sizeof(unsigned int));
    concatenate_rows_kernel_cpu(l.binary_filters, s_concat, l.n, l.size*l.size*l.c);

    unsigned  int* cuda_concat = (unsigned  int *) calloc( m  * k_red, sizeof(unsigned int));
    size_t pull_size = m  * k_red * sizeof(unsigned int);
    cudaError_t status = cudaMemcpy(cuda_concat, l.filters_concat, pull_size, cudaMemcpyDeviceToHost);
    check_error(status);

    for(size_t j = 0; j < m  * k_red; j++){
        if( s_concat[j] !=  cuda_concat[j] ){
            printf("%d, %d, %d \n", j, s_concat[j], cuda_concat[j]);
        }
        assert(s_concat[j] ==  cuda_concat[j]);
    }
    free(s_concat);
    free(cuda_concat);
#endif

    int k_red = k / 32;
    if ( k % 32 > 0 )
        k_red += 1;
    unsigned int* bt;
    cudaError_t status = cudaMalloc((void **)&bt, n * k_red * sizeof(unsigned int) );
    check_error(status);

    for(i = 0; i < l.batch; ++i){

        binarize_xnor_input_gpu(state.input + i*l.inputs, l.c, l.h*l.w, l.mean_input);

        im2col_ongpu(l.mean_input, 1, l.h, l.w,
                   l.size, l.stride, l.pad, l.c_scales);

        fill_ongpu(out_w * out_h, 0, l.c_norm, 1);
        float* kfilters = l.filters_norm;
        gemm_ongpu(0,0,1,n,l.size*l.size,1.,kfilters,l.size*l.size,l.c_scales,n,1.,l.c_norm, n);

#ifdef VIS
        float* c_norm_check =  (float *) calloc( out_w * out_h , sizeof(float));
        cudaError_t status = cudaMemcpy(c_norm_check, l.c_norm,  out_w * out_h  * sizeof(float), cudaMemcpyDeviceToHost);
        check_error(status);
        image im2 = float_to_image(out_w,out_h,1,c_norm_check);
        save_image(im2, "/tmp/img2_cuda.png");
        free(c_norm_check);
#endif

        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c, l.h, l.w, l.size, l.stride, l.pad, l.col_image_gpu);

        float * a = l.filters_gpu;
        float * b = l.col_image_gpu;
        float * c = l.output_gpu;

        xnor_gemm_concat_gpu(l.filters_concat, b, c+i*m*n, m, n, k, a, l.c_norm, bt);

    }
    cuda_free((float *) bt);
    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, state);
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, n);

    activate_array_ongpu(l.output_gpu, m*n*l.batch, l.activation);
    swap_binary(&l);
}

void backward_xnor_conv_layer_gpu(xnor_conv_layer l, network_state state)
{
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = convolutional_out_height(l)*
            convolutional_out_width(l);

    gradient_array_ongpu(l.output_gpu, m*k*l.batch, l.activation, l.delta_gpu);

    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, k);

    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, state);
    }

#ifdef CUDNN
    float one = 1;
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            state.input,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            state.workspace,
            l.workspace_size,
            &one,
            l.dfilterDesc,
            l.filter_updates_gpu);

    if(state.delta){
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.filterDesc,
                l.filters_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                state.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                state.delta);
    }

#else
    int i;
    for(i = 0; i < l.batch; ++i){
        float * a = l.delta_gpu;
        float * b = state.workspace;
        float * c = l.filter_updates_gpu;

        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.workspace);
        gemm_ongpu(0,1,m,n,k,1,a + i*m*k,k,b,k,1,c,n);

        if(state.delta){
            swap_binary(&l);
            float * a = l.filters_gpu;
            float * b = l.delta_gpu;
            float * c = state.workspace;

            gemm_ongpu(1,0,n,k,m,1,a,n,b + i*k*m,k,0,c,k);

            col2im_ongpu(state.workspace, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta + i*l.c*l.h*l.w);
            swap_binary(&l);
        }
    }
#endif
}

void pull_xnor_conv_layer(convolutional_layer layer)
{
    cuda_pull_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_pull_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_pull_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_pull_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
}

void push_xnor_conv_layer(convolutional_layer layer)
{
    cuda_push_array(layer.filters_gpu, layer.filters, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.filter_updates_gpu, layer.filter_updates, layer.c*layer.n*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_push_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_push_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_push_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
}

void update_xnor_conv_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay)
{
    int size = layer.size*layer.size*layer.c*layer.n;

    axpy_ongpu(layer.n, learning_rate/batch, layer.bias_updates_gpu, 1, layer.biases_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.bias_updates_gpu, 1);

    axpy_ongpu(layer.n, learning_rate/batch, layer.scale_updates_gpu, 1, layer.scales_gpu, 1);
    scal_ongpu(layer.n, momentum, layer.scale_updates_gpu, 1);

    axpy_ongpu(size, -decay*batch, layer.filters_gpu, 1, layer.filter_updates_gpu, 1);
    axpy_ongpu(size, learning_rate/batch, layer.filter_updates_gpu, 1, layer.filters_gpu, 1);
    scal_ongpu(size, momentum, layer.filter_updates_gpu, 1);
}



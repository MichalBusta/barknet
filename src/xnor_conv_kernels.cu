#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "convolutional_layer.h"
#include "xnor_conv_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

__global__
void binarize_inp_kernel(float* input, int rows, int cols, int channels)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int r = index / rows;
    int c = index % rows;
    if ((r < rows) && (c < cols))
    {
        //printf("%d %d\n",r, c);
        float mean = 0;
        for( int ch = 0; ch < channels; ch++ ) {
            int idx = c + cols * (r + ch * rows);
            mean += input[idx];
        }
        mean /= channels;
        for( int ch = 0; ch < channels; ch++ ) {
            int idx = c + cols * (r + ch * rows);
            input[idx] = input[idx] > 0 ? mean : -mean;
        }
    }
}



void binarize_input_gpu(float *input, int rows, int cols, int channels)
{

    int x = static_cast<int>(ceilf(static_cast<float>(cols) / BLOCK));
    int y = static_cast<int>(ceilf(static_cast<float>(rows) / BLOCK));

    //printf("x: %d x %d %d\n", rows, cols, BLOCK);

    const dim3 grid (x, y, 1);                                // number of blocks
    const dim3 block(BLOCK, 1, 1);

    binarize_inp_kernel<<<cuda_gridsize(rows * cols), BLOCK>>>(input, rows, cols, channels);
    check_error(cudaPeekAtLastError());
}


void forward_xnor_conv_layer_gpu(xnor_conv_layer l, network_state state)
{
    int i;
    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = convolutional_out_height(l)*
        convolutional_out_width(l);

    fill_ongpu(l.outputs*l.batch, 0, l.output_gpu, 1);

    binarize_filters_gpu(l.filters_gpu, l.n, l.c*l.size*l.size, l.binary_filters_gpu);
    swap_binary(&l);

    for(i = 0; i < l.batch; ++i){

        binarize_input_gpu(state.input + i*l.c*l.h*l.w, l.h, l.w, l.c);

        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, l.col_image_gpu);

        float * a = l.filters_gpu;
        float * b = l.col_image_gpu;
        float * c = l.output_gpu;
        gemm_ongpu(0,0,m,n,k,1.,a,k,b,n,1.,c+i*m*n,n);
    }

    if (l.batch_normalize) {
        if (state.train) {
            fast_mean_gpu(l.output_gpu, l.batch, l.n, l.out_h*l.out_w, l.mean_gpu);
            fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.n, l.out_h*l.out_w, l.variance_gpu);

            scal_ongpu(l.n, .95, l.rolling_mean_gpu, 1);
            axpy_ongpu(l.n, .05, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
            scal_ongpu(l.n, .95, l.rolling_variance_gpu, 1);
            axpy_ongpu(l.n, .05, l.variance_gpu, 1, l.rolling_variance_gpu, 1);

            copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
            normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.n, l.out_h*l.out_w);
            copy_ongpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);
        } else {
            normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.n, l.out_h*l.out_w);
        }

        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.n, l.out_h*l.out_w);
    }
    add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, n);

    activate_array_ongpu(l.output_gpu, m*n*l.batch, l.activation);
    if(l.dot > 0) dot_error_gpu(l);
    swap_binary(&l);
}

void backward_xnor_conv_layer_gpu(xnor_conv_layer l, network_state state)
{
    int i;
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = convolutional_out_height(l)*
        convolutional_out_width(l);

    gradient_array_ongpu(l.output_gpu, m*k*l.batch, l.activation, l.delta_gpu);

    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, k);

    if(l.batch_normalize){
        backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h, l.scale_updates_gpu);

        scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.n, l.out_h*l.out_w);

        fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.n, l.out_w*l.out_h, l.mean_delta_gpu);
        fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.n, l.out_w*l.out_h, l.variance_delta_gpu);
        normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.n, l.out_w*l.out_h, l.delta_gpu);
    }

    for(i = 0; i < l.batch; ++i){
        float * a = l.delta_gpu;
        float * b = l.col_image_gpu;
        float * c = l.filter_updates_gpu;

        im2col_ongpu(state.input + i*l.c*l.h*l.w, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, l.col_image_gpu);
        gemm_ongpu(0,1,m,n,k,1,a + i*m*k,k,b,k,1,c,n);

        if(state.delta){
            swap_binary(&l);
            float * a = l.filters_gpu;
            float * b = l.delta_gpu;
            float * c = l.col_image_gpu;

            gemm_ongpu(1,0,n,k,m,1,a,n,b + i*k*m,k,0,c,k);

            col2im_ongpu(l.col_image_gpu, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta + i*l.c*l.h*l.w);
            swap_binary(&l);
        }
    }
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



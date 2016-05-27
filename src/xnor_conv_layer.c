#include "utils.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <assert.h>
#include "convolutional_layer.h"
#include "xnor_conv_layer.h"
#include "xnor_gemm.h"

#pragma GCC push_options
#pragma GCC optimize ("O0")


xnor_conv_layer make_xnor_conv_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation, int batch_normalize, int binary)
{
    int i;
    xnor_conv_layer l = {0};
    l.type = XNOR_CONV;

    l.h = h;
    l.w = w;
    l.c = c;
    l.n = n;
    l.binary = binary;
    l.batch = batch;
    l.stride = stride;
    l.size = size;
    l.pad = pad;
    l.batch_normalize = batch_normalize;

    l.filters = calloc(c*n*size*size, sizeof(float));
    l.filter_updates = calloc(c*n*size*size, sizeof(float));
    l.a_norm = calloc(w * h, sizeof(float));

    l.biases = calloc(n, sizeof(float));
    l.bias_updates = calloc(n, sizeof(float));

    // float scale = 1./sqrt(size*size*c);
    float scale = sqrt(2./(size*size*c));
    for(i = 0; i < c*n*size*size; ++i) l.filters[i] = scale*rand_uniform(-1, 1);
    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    l.out_h = out_h;
    l.out_w = out_w;
    l.out_c = n;
    l.outputs = l.out_h * l.out_w * l.out_c;
    l.inputs = l.w * l.h * l.c;

    l.col_image = calloc(out_h*out_w*size*size*c, sizeof(float));
    l.output = calloc(l.batch*out_h * out_w * n, sizeof(float));
    l.delta  = calloc(l.batch*out_h * out_w * n, sizeof(float));


    l.binary_filters = calloc(c*n*size*size, sizeof(float));
    l.cfilters = calloc(c*n*size*size, sizeof(char));
    l.scales = calloc(n, sizeof(float));

    int k = l.size*l.size*l.c;
    int k_red = k / 32;
    if( k > k_red * 32 )
        k_red += 1;

    l.scales = calloc(n, sizeof(float));
    if(batch_normalize){
        l.scale_updates = calloc(n, sizeof(float));
        for(i = 0; i < n; ++i){
            l.scales[i] = 1;
        }

        l.mean = calloc(n, sizeof(float));
        l.variance = calloc(n, sizeof(float));

        l.rolling_mean = calloc(n, sizeof(float));
        l.rolling_variance = calloc(n, sizeof(float));
    }

    int kpad = l.size / 2;
#ifdef GPU
    l.filters_gpu = cuda_make_array(l.filters, c*n*size*size);
    l.filter_updates_gpu = cuda_make_array(l.filter_updates, c*n*size*size);

    l.biases_gpu = cuda_make_array(l.biases, n);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

    l.scales_gpu = cuda_make_array(l.scales, n);
    l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

    l.col_image_gpu = cuda_make_array(l.col_image, out_h*out_w*size*size*c);
    l.delta_gpu = cuda_make_array(l.delta, l.batch*out_h*out_w*n);
    l.output_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);

    l.binary_filters_gpu = cuda_make_array(l.filters, c*n*size*size);

    //xnor
    cudaError_t status = cudaMalloc((void **)&l.mean_input, ( l.h + 2 *  kpad )*(l.w + 2 * kpad) * sizeof(float));
    check_error(status);
    status = cudaMalloc((void **)&l.c_scales, out_h * out_w * size* size * sizeof(float));
    check_error(status);
    status = cudaMalloc((void **)&l.c_norm, out_w * out_h  * sizeof(float));
    check_error(status);
    status = cudaMalloc((void **)&l.filters_norm, n*l.size*l.size*sizeof(float));
    check_error(status);
    float fact = 1.0f / (l.size*l.size);
    fill_ongpu(n*l.size*l.size, fact, l.filters_norm, 1);
    status = cudaMalloc((void **)&l.filters_concat, n  * k_red * sizeof(unsigned int));
    check_error(status);

    //xnor

    if(batch_normalize){
        l.mean_gpu = cuda_make_array(l.mean, n);
        l.variance_gpu = cuda_make_array(l.variance, n);

        l.rolling_mean_gpu = cuda_make_array(l.mean, n);
        l.rolling_variance_gpu = cuda_make_array(l.variance, n);

        l.mean_delta_gpu = cuda_make_array(l.mean, n);
        l.variance_delta_gpu = cuda_make_array(l.variance, n);

        l.x_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
        l.x_norm_gpu = cuda_make_array(l.output, l.batch*out_h*out_w*n);
    }
#else
    l.mean_input = calloc(( l.h + 2 *  kpad ) * (l.w + 2 * kpad), sizeof(float));
    l.c_scales = malloc(out_h * out_w * size* size *sizeof(float));
    l.c_norm= calloc(out_w * out_h, sizeof(float));

    l.filters_norm = malloc(n*l.size*l.size*sizeof(float));
    float fact = 1.0f / (l.size*l.size);
    for(int i = 0; i < n*l.size*l.size; i++){
        l.filters_norm[i] = fact;
    }

    l.filters_concat = calloc( n  * k_red, sizeof(unsigned int));

#endif
    l.activation = activation;

    fprintf(stderr, "X-NOR Convolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image\n", h,w,c,n, out_h, out_w, n);

    return l;
}

//#define VERBOSE_T
//#define NUMERIC_TEST 1
//#define VIS 1

void forward_xnor_conv_layer(xnor_conv_layer l, network_state state)
{

#ifdef VERBOSE_T
    clock_t time=clock();
    clock_t t_gemm = 0;
#endif

    int out_h = convolutional_out_height(l);
    int out_w = convolutional_out_width(l);
    int i;

    fill_cpu(l.outputs*l.batch, 0, l.output, 1);

    if(state.train) {
        binarize_filters(l.filters, l.n, l.c * l.size * l.size, l.binary_filters);
        swap_binary(&l);
        concatenate_rows_kernel_cpu(l.filters, l.filters_concat, l.n, l.size*l.size*l.c);
    }

    int m = l.n;
    int k = l.size*l.size*l.c;
    int n = out_h*out_w;

    float *a = l.filters;
    float *b = l.col_image;
    float *c = l.output;

    for(i = 0; i < l.batch; ++i){
        #pragma omp parallel for
        for(int row = 0; row < l.h; row++){
            for(int col = 0; col < l.w; col++){
                float mean = 0;
                for(int c = 0; c < l.c; c++){
                    mean += fabs(state.input[(row + c * l.h) * l.w + col]);
                }
                mean = mean / l.c;
                assert(mean <= 1);
                l.mean_input[row * l.w + col] = mean;
                if(state.train) {
                    for (int c = 0; c < l.c; c++) {
                        state.input[(row + c * l.h) * l.w + col] = state.input[(row + c * l.h) * l.w + col] > 0 ? mean : -mean;
                    }
                }
            }
        }

        im2col_cpu(l.mean_input, 1, l.h, l.w,
                   l.size, l.stride, l.pad, l.c_scales);

#ifdef VIS
        image im = float_to_image(out_w, out_h, 1, l.c_scales);
        image im_v = float_to_image(l.w,l.h, 1, l.mean_input);
        save_image(im, "/tmp/img.png");
        save_image(im_v, "/tmp/img1.png");

#endif
        fill_cpu(out_w * out_h, 0, l.c_norm, 1);

        float* kfilters = l.filters_norm;
        gemm(0,0,1,n,l.size*l.size,1,kfilters,l.size*l.size,l.c_scales,n,1,l.c_norm,n);

#ifdef VIS
        image im2 = float_to_image(out_w,out_h,1,l.c_norm);
        save_image(im2, "/tmp/img2.png");
#endif

        im2col_cpu(state.input, l.c, l.h, l.w,
                   l.size, l.stride, l.pad, b);



#ifdef VERBOSE_T
        clock_t timeg=clock();
#endif
        //gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
        xnor_gemm_concat(l.filters_concat, b, c, m, n, k);

        for(int ii = 0; ii < m; ii++){
            register float A_PART = fabs(a[ii*k]);
            for(int j = 0; j < n; j++) {
                c[ii * n + j] *= (fabs(l.c_norm[j]) *  A_PART);
                assert(!isinf(c[ii * n + j]));
            }
        }


#ifdef NUMERIC_TEST
        //float* c_check = calloc( m  * n, sizeof(float));
        float* c_check2 = calloc( m  * n, sizeof(float));
        gemm(0,0,m,n,k,1,a,k,b,n,1,c_check2,n);
        //gemm_bin2(m, n, k, a, k, b, n, c_check, n);


        for(int i = 0; i < m; i++ ){
            for(int j = 0; j < n; j++) {

                if( fabs(c[i * n + j] - c_check2[i * n + j] ) > 1e-4){
                    printf("%d %d, %f, %f\n", i, j, c[i * n + j], c_check2[i * n + j]);
                }
                //assert(c_check[i * n + j] == c[i * n + j]);
            }
        }

        image im3t = float_to_image(out_w,out_h,1,c_check2);
        save_image(im3t, "/tmp/img3t.png");

#endif

#ifdef VERBOSE_T
        t_gemm += clock()-timeg;
#endif

#ifdef VIS
        image im3 = float_to_image(out_w,out_h,1,c);
        save_image(im3, "/tmp/img3.png");
#endif

        c += n*m;
        state.input += l.c*l.h*l.w;

    }

    if(l.batch_normalize){
        if(state.train){
            mean_cpu(l.output, l.batch, l.n, l.out_h*l.out_w, l.mean);
            variance_cpu(l.output, l.mean, l.batch, l.n, l.out_h*l.out_w, l.variance);
            normalize_cpu(l.output, l.mean, l.variance, l.batch, l.n, l.out_h*l.out_w);
        } else {
            normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.n, l.out_h*l.out_w);
        }
        scale_bias(l.output, l.scales, l.batch, l.n, out_h*out_w);
    }



    add_bias(l.output, l.biases, l.batch, l.n, out_h*out_w);
    activate_array(l.output, m*n*l.batch, l.activation);
    if(state.train || 1) {
        swap_binary(&l);
    }

#ifdef VERBOSE_T
    printf("XCO: %lf / %lf sec\n", sec(clock()-time), sec(t_gemm));
#endif
}

void lgradient_array(const float *x, const int n, const ACTIVATION a, float *delta)
{
    int i;
    for(i = 0; i < n; ++i){
        float xx = fmax(-1, fmin(1, x[i]));
        delta[i] *= gradient(xx, a);
        assert(!isnan(delta[i]));
    }
}

void backward_xnor_conv_layer(xnor_conv_layer l, network_state state)
{
    int i;
    int m = l.n;
    int n = l.size*l.size*l.c;
    int k = convolutional_out_height(l)*
            convolutional_out_width(l);

    gradient_array(l.output, m*k*l.batch, l.activation, l.delta);
    backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);

    for(i = 0; i < l.batch; ++i){
        float *a = l.delta + i*m*k;
        float *b = l.col_image;
        float *c = l.filter_updates;

        float *im = state.input+i*l.c*l.h*l.w;

        im2col_cpu(im, l.c, l.h, l.w,
                   l.size, l.stride, l.pad, b);
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

        if(state.delta){
            swap_binary(&l);
            a = l.filters;
            b = l.delta + i*m*k;
            c = l.col_image;

            gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);

            col2im_cpu(l.col_image, l.c,  l.h,  l.w,  l.size,  l.stride, l.pad, state.delta+i*l.c*l.h*l.w);
            swap_binary(&l);
        }
    }
}

void update_xnor_conv_layer(xnor_conv_layer l, int batch, float learning_rate, float momentum, float decay)
{
    int size = l.size*l.size*l.c*l.n;
    axpy_cpu(l.n, learning_rate/batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, momentum, l.bias_updates, 1);

    axpy_cpu(size, -decay*batch, l.filters, 1, l.filter_updates, 1);
    axpy_cpu(size, learning_rate/batch, l.filter_updates, 1, l.filters, 1);
    scal_cpu(size, momentum, l.filter_updates, 1);
}

void resize_xnor_conv_layer(xnor_conv_layer *l, int w, int h)
{
	l->w = w;
	l->h = h;
	int out_w = convolutional_out_width(*l);
	int out_h = convolutional_out_height(*l);

	l->out_w = out_w;
	l->out_h = out_h;

	l->outputs = l->out_h * l->out_w * l->out_c;
	l->inputs = l->w * l->h * l->c;

	l->col_image = realloc(l->col_image,
			out_h*out_w*l->size*l->size*l->c*sizeof(float));
	l->output = realloc(l->output,
			l->batch*out_h * out_w * l->n*sizeof(float));
	l->delta  = realloc(l->delta,
			l->batch*out_h * out_w * l->n*sizeof(float));

#ifdef GPU
	cuda_free(l->col_image_gpu);
	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);

	l->col_image_gpu = cuda_make_array(l->col_image, out_h*out_w*l->size*l->size*l->c);
	l->delta_gpu =     cuda_make_array(l->delta, l->batch*out_h*out_w*l->n);
	l->output_gpu =    cuda_make_array(l->output, l->batch*out_h*out_w*l->n);
#endif
}

void load_xnor_conv_weights(layer l, FILE *fp)
{
    int num = l.n*l.c*l.size*l.size;
    fread(l.biases, sizeof(float), l.n, fp);
    if (l.batch_normalize && (!l.dontloadscales)){
        fread(l.scales, sizeof(float), l.n, fp);
        fread(l.rolling_mean, sizeof(float), l.n, fp);
        fread(l.rolling_variance, sizeof(float), l.n, fp);
    }
    fread(l.filters, sizeof(float), num, fp);
    if (l.flipped) {
        transpose_matrix(l.filters, l.c*l.size*l.size, l.n);
    }

#ifdef GPU
    if(gpu_index >= 0){
        push_convolutional_layer(l);
    }
    binarize_filters_gpu(l.filters_gpu, l.n, l.c*l.size*l.size, l.binary_filters_gpu);
    concatenate_rows_gpu(l.binary_filters_gpu, l.filters_concat, l.n, l.size*l.size*l.c);

#else

    binarize_filters(l.filters, l.n, l.c * l.size * l.size, l.binary_filters);
    swap_binary(&l);
    concatenate_rows_kernel_cpu(l.filters, l.filters_concat, l.n, l.size*l.size*l.c);
    swap_binary(&l);
#endif
}

#pragma GCC pop_options

#ifndef CONV_XNOR_LAYER_H
#define CONV_XNOR_LAYER_H

#include "cuda.h"
#include "image.h"
#include "activations.h"
#include "layer.h"
#include "network.h"

typedef layer xnor_conv_layer;

#ifdef GPU

void forward_xnor_conv_layer_gpu(xnor_conv_layer l, network_state state);
void backward_xnor_conv_layer_gpu(xnor_conv_layer l, network_state state);
void update_xnor_conv_layer_gpu(convolutional_layer layer, int batch, float learning_rate, float momentum, float decay);

void concatenate_rows_gpu(float *a, unsigned int *b, int rows, int cols);

#endif

xnor_conv_layer make_xnor_conv_layer(int batch, int h, int w, int c, int n, int size, int stride, int pad, ACTIVATION activation, int batch_normalization, int binary);
void forward_xnor_conv_layer(const xnor_conv_layer layer, network_state state);
void update_xnor_conv_layer(xnor_conv_layer layer, int batch, float learning_rate, float momentum, float decay);

void backward_xnor_conv_layer(xnor_conv_layer layer, network_state state);

void resize_xnor_conv_layer(xnor_conv_layer *layer, int w, int h);

void load_xnor_conv_weights(layer l, FILE *fp);

#endif


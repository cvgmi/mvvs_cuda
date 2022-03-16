#include <torch/extension.h>

#include <vector>
#include <iostream>

// number of samples per voxel. TODO: Consider how to make this dynamic
#define PER_VOXEL 45

// source: https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor mvc_cuda_log_forward(torch::Tensor x,
                                   torch::Tensor weights);

torch::Tensor mvc_cuda_exp_forward(torch::Tensor x,
                                   torch::Tensor m);

torch::Tensor mvc_forward_log(torch::Tensor x, 
                  torch::Tensor weights) {

    // MVC Forward Pass:
    // x: [batches, channels_in, H, W, D, N]
    // weights: [channels_out, kernel_size, kernel_size, kernel_size, channels_in]

    // basic checks
    CHECK_INPUT(x);
    CHECK_INPUT(weights);

    // assert that kernel size dimensions are all the same
    TORCH_CHECK(weights.size(1) == weights.size(2) && weights.size(2) == weights.size(3));
    TORCH_CHECK(x.size(5) == PER_VOXEL);
    TORCH_CHECK(weights.size(1) % 2 == 1);

    return mvc_cuda_log_forward(x, weights);
}

torch::Tensor mvc_forward_exp(torch::Tensor x, torch::Tensor m){

    // MVC Forward Pass:
    // x: [batches, channels_in, H, W, D, N]
    // weights: [channels_out, kernel_size, kernel_size, kernel_size, channels_in]

    // basic checks
    CHECK_INPUT(x);
    CHECK_INPUT(m);

    // assert that kernel size dimensions are all the same
    TORCH_CHECK(x.size(5) == PER_VOXEL);
    TORCH_CHECK(m.size(5) == PER_VOXEL);

    return mvc_cuda_exp_forward(x, m);
}

std::vector<torch::Tensor> mvc_cuda_exp_backward(torch::Tensor x, torch::Tensor m, torch::Tensor d_exp);
torch::Tensor mvc_cuda_log_backward(torch::Tensor x, torch::Tensor weights, torch::Tensor d_log);
torch::Tensor mvc_cuda_log_weights_backward(torch::Tensor x, torch::Tensor weights, torch::Tensor d_log);

std::vector<torch::Tensor> mvc_backward_exp(torch::Tensor x, torch::Tensor m, torch::Tensor d_exp){
    CHECK_INPUT(x);
    CHECK_INPUT(m);
    CHECK_INPUT(d_exp);

    return mvc_cuda_exp_backward(x, m, d_exp);
}

torch::Tensor mvc_backward_log(torch::Tensor x, torch::Tensor weights, torch::Tensor d_log){
    CHECK_INPUT(x);
    CHECK_INPUT(weights);
    CHECK_INPUT(d_log);

    return mvc_cuda_log_backward(x, weights, d_log);
}

torch::Tensor mvc_backward_log_weights(torch::Tensor x, torch::Tensor weights, torch::Tensor d_log){
    CHECK_INPUT(x);
    CHECK_INPUT(weights);
    CHECK_INPUT(d_log);

    return mvc_cuda_log_weights_backward(x, weights, d_log);
}

//TESTING:

std::vector<torch::Tensor> exp_cuda_backward(torch::Tensor X, 
                                torch::Tensor d_Exp,
                                int batch, 
                                int c, int mid_c, 
                                int x, int y, int z,
                                int x_m, int y_m, int z_m);

std::vector<torch::Tensor> mvc_exp_backward(torch::Tensor X, 
                                torch::Tensor d_Exp,
                                int batch, 
                                int c, int mid_c, 
                                int x, int y, int z,
                                int x_m, int y_m, int z_m){
    TORCH_CHECK(X.size(5) == PER_VOXEL);
    return exp_cuda_backward(X, d_Exp, batch, c, mid_c, x, y, z, x_m, y_m, z_m);
}



std::vector<torch::Tensor> log_cuda_backward(torch::Tensor X, 
                                torch::Tensor d_Log,
                                int batch, 
                                int c, int mid_c, 
                                int x, int y, int z,
                                int x_m, int y_m, int z_m);


std::vector<torch::Tensor> mvc_log_backward(torch::Tensor X, 
                                torch::Tensor d_Log,
                                int batch, 
                                int c, int mid_c, 
                                int x, int y, int z,
                                int x_m, int y_m, int z_m){
    TORCH_CHECK(X.size(5) == PER_VOXEL);
    return log_cuda_backward(X, d_Log, batch, c, mid_c, x, y, z, x_m, y_m, z_m);
}
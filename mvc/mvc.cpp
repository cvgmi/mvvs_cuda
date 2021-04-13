#include <torch/extension.h>

#include<vector>

// number of samples per voxel. TODO: Consider how to make this dynamic
#define PER_VOXEL 45

// source: https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor mvc_cuda_forward(torch::Tensor x,
                               torch::Tensor weights);

torch::Tensor mvc(torch::Tensor x, 
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

    return mvc_cuda_forward(x, weights);
}

torch::Tensor d_mvc(torch::Tensor x,
                    torch::Tensor weights) {
        auto s = torch::sigmoid(x);
        return (1 - s) * s;
}


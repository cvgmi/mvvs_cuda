#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <math.h>

#include <vector>
#include <iostream>

// size of the edge of a block 
#define BLOCK_SIZE 6
#define BLOCK_SIZE_TILED 4
// number of samples per voxel. TODO: Consider how to make this dynamic
#define PER_VOXEL 45
// epsilon
#define EPS 0.000000001

__device__ dim3 offset2coordinates_RM_exp_bw(size_t offset, 
                                      size_t height,
                                      size_t width,
                                      size_t depth){

    const int block_x = offset % height;
    offset = offset / height;
    const int block_y = offset % width;
    offset = offset / width;
    const int block_z = offset % depth;
    
    dim3 output(block_x, block_y, block_z);
    return output;
}

template <typename scalar_t, typename accessor>
__device__ void d_mvcExp(
        accessor X,
        accessor M,
        torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> d_Exp,
        torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> d_X,
        torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> d_M,
        int batch, 
        int c, int mid_c,
        int x_m, int y_m, int z_m,
        int x, int y, int z){
    // Compute derivatire of loss w.r.t mvcExp inputs, given derivative of loss w.r.t mvcExp outputs.
    // check with matrixcalculus.org: cos(norm2(v))*m + sin(norm2(v))*v/norm2(v)

    scalar_t norm = 0;
    for(int n = 0; n < PER_VOXEL; n++)
        norm += X[batch][c][x][y][z][n]*X[batch][c][x][y][z][n];

    scalar_t inner_dExp_X = 0;
    for(int n = 0; n < PER_VOXEL; n++)
        inner_dExp_X += X[batch][c][x][y][z][n]*d_Exp[batch][c][x][y][z][n];
    
    scalar_t inner_dExp_m = 0;
    for(int n = 0; n < PER_VOXEL; n++)
        inner_dExp_m += M[batch][mid_c][x_m][y_m][z_m][n]*d_Exp[batch][c][x][y][z][n];


    norm = sqrt(norm);
    if(norm == 0)
        norm = 1;
    
    // take weighted combination 
    scalar_t cos_norm = cos(norm);
    scalar_t sin_norm = sin(norm)/norm;

    for(int n = 0; n < PER_VOXEL; n++){

        // compute d_X:
        // cos(t0)/t0^2 T2
        d_X[batch][c][x][y][z][n] = (cos_norm/pow(norm, 2))*X[batch][c][x][y][z][n]*inner_dExp_X;
        // - t3 m Transpose[v]
        d_X[batch][c][x][y][z][n] -= sin_norm*X[batch][c][x][y][z][n]*inner_dExp_m;
        // - t1/t0^3 T2
        d_X[batch][c][x][y][z][n] -= (sin_norm/pow(norm, 2))*X[batch][c][x][y][z][n]*inner_dExp_X;
        // + t3 I
        d_X[batch][c][x][y][z][n] += d_Exp[batch][c][x][y][z][n]*sin_norm;

        // compute d_M:
        d_M[batch][mid_c][x_m][y_m][z_m][n] += d_Exp[batch][c][x][y][z][n]*cos_norm;
    }
}

template <typename scalar_t>
__global__ void mvc_cuda_exp_backward_kernel(
            torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> X,
            torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> M,
            torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> d_exp,
            torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> d_X,
            torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> d_M,
            dim3 block_grid_dims){
    // compute block coordinates from linearized blockIdx.x
    size_t batch = blockIdx.x;
    size_t block_offset = blockIdx.y;

    const dim3 blockCoord = offset2coordinates_RM_exp_bw(block_offset, block_grid_dims.x, block_grid_dims.y, block_grid_dims.z);
    const int kernel_size = M.size(2)-X.size(2)+1;

    // kernel size must be odd
    const int half_kernel = (kernel_size+1)/2;
    const int half_kernel_floor = (kernel_size)/2;

    // compute output element coordinate that we are going to compute
    const size_t x = blockCoord.x*blockDim.x + threadIdx.x + half_kernel_floor;
    const size_t y = blockCoord.y*blockDim.y + threadIdx.y + half_kernel_floor;
    const size_t z = blockCoord.z*blockDim.z + threadIdx.z + half_kernel_floor;
    const size_t mid_channel = M.size(1)/2;

    // check we are within bounds
    if(x < 0 || x - half_kernel_floor >= d_X.size(2) ||
       y < 0 || y - half_kernel_floor >= d_X.size(3) ||
       z < 0 || z - half_kernel_floor >= d_X.size(4))
        return;

    for(int c_out = 0; c_out < X.size(1); c_out++){
        d_mvcExp(X, M, d_exp, d_X, d_M,
                batch, c_out, mid_channel,
                x, y, z,
                x-half_kernel_floor, y-half_kernel_floor, z-half_kernel_floor);
    }
}

std::vector<torch::Tensor> mvc_cuda_exp_backward(torch::Tensor x,
                                    torch::Tensor M,
                                    torch::Tensor d_exp){
    // MVC Forward Pass:
    // x: [batches, channels_in, H, W, D, N]
    // m: [batches, channels_in, H_og, W_og, D_og, N]

    auto H_out = x.size(2);
    auto W_out = x.size(3);
    auto D_out = x.size(4);

    // allocate output of size [batches, channels_out, H_out, W_out, D_out, N]
    torch::Tensor d_X = torch::zeros({x.size(0), x.size(1), x.size(2), x.size(3), x.size(4), x.size(5)},
                                        torch::dtype(x.dtype()).device(torch::kCUDA));

    torch::Tensor d_M = torch::zeros({M.size(0), M.size(1), M.size(2), M.size(3), M.size(4), M.size(5)},
                                        torch::dtype(x.dtype()).device(torch::kCUDA));
    
    // call kernels using ATEN dispatch macro
    // CUDA execution configuration:
    // block grid size (with ceiling division)
    int block_grid_h = (H_out+BLOCK_SIZE-1)/BLOCK_SIZE;
    int block_grid_w = (W_out+BLOCK_SIZE-1)/BLOCK_SIZE;
    int block_grid_d = (D_out+BLOCK_SIZE-1)/BLOCK_SIZE;
    const dim3 block_grid_dims(block_grid_h, block_grid_w, block_grid_d);
    int block_grid = block_grid_h*block_grid_w*block_grid_d;
    // blocks: [batches, output_size/block_size]
    const dim3 blocks(x.size(0), block_grid);
    // threads: block_size
    const dim3 threads(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "mvc_cuda_exp_backward_kernel", ([&] {
                mvc_cuda_exp_backward_kernel<scalar_t><<<blocks, threads>>>(
                    x.packed_accessor<scalar_t, 6, torch::RestrictPtrTraits, size_t>(),
                    M.packed_accessor<scalar_t, 6, torch::RestrictPtrTraits, size_t>(),
                    d_exp.packed_accessor<scalar_t, 6, torch::RestrictPtrTraits, size_t>(),
                    d_X.packed_accessor<scalar_t, 6, torch::RestrictPtrTraits, size_t>(),
                    d_M.packed_accessor<scalar_t, 6, torch::RestrictPtrTraits, size_t>(),
                    block_grid_dims);
    }));

    return {d_X, d_M};
}

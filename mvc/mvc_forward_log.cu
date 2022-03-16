#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include <math.h>

#include <vector>

// size of the edge of a block 
#define BLOCK_SIZE 6
#define BLOCK_SIZE_TILED 4
// number of samples per voxel. TODO: Consider how to make this dynamic
#define PER_VOXEL 45
// epsilon
#define EPS 1e-6

__device__ dim3 offset2coordinates_RM_log(size_t offset, 
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
__host__ __device__ void mvcLog(
        accessor X,
        scalar_t* output,
        int batch, 
        int c, int mid_c,
        int x, int y, int z,
        int x_m, int y_m, int z_m){

    // SOURCE: https://projects.csail.mit.edu/atemuri/wiki/images/f/fe/SrivastavaJermynJoshiCVPR2007.pdf eq. 9
    // compute inner product between X[x, y, z] and X[x_m, y_m, z_m]
    scalar_t inner = 0;
    for(int n = 0; n < PER_VOXEL; n++)
        inner += (X[batch][c][x][y][z][n])*(X[batch][mid_c][x_m][y_m][z_m][n]);
    

    // compute vector u 
    for(int n = 0; n < PER_VOXEL; n++){
        scalar_t X_n = X[batch][c][x][y][z][n];
        scalar_t X_m_n = X[batch][mid_c][x_m][y_m][z_m][n];

        scalar_t u = X_n - inner*X_m_n;
        output[n] = u;
    }
    
    // compute the norm of the vector u
    scalar_t u_norm = 0;
    for(int n = 0; n < PER_VOXEL; n++)
        u_norm += output[n]*output[n];
    

    // to avoid floating point innacuracies leading to undefined acos below
    if(inner > 1)
        inner = 1;
    else if(inner < -1)
        inner = -1;

    // write output
    u_norm = sqrt(u_norm);
    for(int n = 0; n < PER_VOXEL; n++){
        if (abs(u_norm) > EPS)
            output[n] = output[n]*acos(inner)/u_norm;
        else{
            output[n] = 0;
        }
    }
}

template <typename scalar_t>
__global__ void mvc_cuda_log_forward_kernel(
            torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> X,
            const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits, size_t> __restrict__ weights,
            torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> output,
            dim3 block_grid_dims){
    // compute block coordinates from linearized blockIdx.x
    size_t batch = blockIdx.x;
    size_t block_offset = blockIdx.y;

    const dim3 blockCoord = offset2coordinates_RM_log(block_offset, block_grid_dims.x, block_grid_dims.y, block_grid_dims.z);

    const int kernel_size = weights.size(1);

    // kernel size must be odd
    const int half_kernel = (kernel_size+1)/2;
    const int half_kernel_floor = (kernel_size)/2;

    // compute output element coordinate that we are going to compute
    const size_t x = blockCoord.x*blockDim.x + threadIdx.x + half_kernel_floor;
    const size_t y = blockCoord.y*blockDim.y + threadIdx.y + half_kernel_floor;
    const size_t z = blockCoord.z*blockDim.z + threadIdx.z + half_kernel_floor;

    const size_t mid_channel = X.size(1)/2;



    // check we are within bounds
    if(x < 0 || x - half_kernel_floor >= output.size(2) ||
       y < 0 || y - half_kernel_floor >= output.size(3) ||
       z < 0 || z - half_kernel_floor >= output.size(4))
        return;

    // iterate over channels and spatial dimensions
    scalar_t mvc_out[PER_VOXEL];
    for(int c = 0; c < X.size(1); c++){
        for(int i = -half_kernel+1; i < half_kernel; i++){
            for(int j = -half_kernel+1; j < half_kernel; j++){
                for(int k = -half_kernel+1; k < half_kernel; k++){
                    int x_o = x + i;
                    int y_o = y + j;
                    int z_o = z + k;

                    // check we are within the image
                    if(0 <= x_o && x_o < X.size(2) &&
                       0 <= y_o && y_o < X.size(3) &&
                       0 <= z_o && z_o < X.size(4)){
                        // Compute Log of X[batch, c, x_o, y_o, z_o] based at X[batch, c, x, y, z] and write it to
                        // output

                        // TODO: we do not need to compute this when x_o,y_o,z_o = x,y,z

                        // write out weighted Log to output array
                        for(int c_out = 0; c_out < weights.size(0); c_out++){
                            mvcLog(X, mvc_out,
                                batch, c, c_out % X.size(1),
                                x_o, y_o, z_o,
                                x, y, z);

                            scalar_t weight = weights[c_out][i+half_kernel-1][j+half_kernel-1][k+half_kernel-1][c];

                            for(int n = 0; n < X.size(5); n++){
                                output[batch][c_out][x-half_kernel_floor][y-half_kernel_floor][z-half_kernel_floor][n] += mvc_out[n]*weight;
                            }
                        }
                    }
                }
            }
        }
    }
}

torch::Tensor mvc_cuda_log_forward(torch::Tensor x,
                               torch::Tensor weights){
    // MVC Forward Pass:
    // x: [batches, channels_in, H, W, D, N]
    // weights: [channels_out, kernel_size, kernel_size, kernel_size, channels_in]

    auto H_out = x.size(2)-(weights.size(1))+1;
    auto W_out = x.size(3)-(weights.size(1))+1;
    auto D_out = x.size(4)-(weights.size(1))+1;

    // allocate output of size [batches, channels_out, H_out, W_out, D_out, N]
    torch::Tensor output = torch::zeros({x.size(0), weights.size(0), H_out, W_out, D_out, x.size(5)},
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

    AT_DISPATCH_FLOATING_TYPES(x.type(), "mvc_cuda_log_forward_kernel", ([&] {
                mvc_cuda_log_forward_kernel<scalar_t><<<blocks, threads>>>(
                    x.packed_accessor<scalar_t, 6, torch::RestrictPtrTraits, size_t>(),
                    weights.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits, size_t>(),
                    output.packed_accessor<scalar_t, 6, torch::RestrictPtrTraits, size_t>(),
                    block_grid_dims);
    }));

    return output;
}

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
#define EPS 1e-6

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

__device__ dim3 offset2coordinates_RM_log_bw(size_t offset, 
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
__device__ void d_mvcLog(
        torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> X,
        accessor d_Log,
        torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> d_X,
        torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> d_M,
        const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits, size_t> weights,
        int batch, 
        int c, int mid_c,
        int x, int y, int z,
        int x_m, int y_m, int z_m,
        int half_kernel, int out_c,
        int i, int j, int k){
    
    // Compute derivative of loss w.r.t mvcLog inputs, given derivative of loss w.r.t mvcLog outputs.
    // check with: matrixcalculus.org: (v-m*sum(v .* m))*arccos(sum(v .* m))/norm2(v-m*sum(v .* m))

    scalar_t inner = 0;
    for(int n = 0; n < PER_VOXEL; n++)
        inner += (X[batch][c][x][y][z][n])*(X[batch][mid_c][x_m][y_m][z_m][n]);

    // to avoid floating point innacuracies leading to undefined acos below
    if(inner > 1)
        inner = 1;
    else if(inner < -1)
        inner = -1;

    // compute vector u 
    scalar_t t2 = 0;
    for(int n = 0; n < PER_VOXEL; n++){
        t2 += (X[batch][c][x][y][z][n]-inner*X[batch][mid_c][x_m][y_m][z_m][n])*(X[batch][c][x][y][z][n]-inner*X[batch][mid_c][x_m][y_m][z_m][n]);
    }

    // inner product of d_Log and t1
    scalar_t inner_dLog_t1 = 0;
    for(int n = 0; n < PER_VOXEL; n++){
        inner_dLog_t1 += (d_Log[batch][out_c][x_m-half_kernel][y_m-half_kernel][z_m-half_kernel][n])*(X[batch][c][x][y][z][n]-inner*X[batch][mid_c][x_m][y_m][z_m][n]);
    }

    // inner prodcut of middle point and t1
    scalar_t inner_m_t1 = 0;
    for(int n = 0; n < PER_VOXEL; n++){
        inner_m_t1 += (X[batch][mid_c][x_m][y_m][z_m][n])*(X[batch][c][x][y][z][n]-inner*X[batch][mid_c][x_m][y_m][z_m][n]);
    }

    // inner product of middle point and d_Log
    scalar_t inner_dLog_m = 0;
    for(int n = 0; n < PER_VOXEL; n++){
        inner_dLog_m += (X[batch][mid_c][x_m][y_m][z_m][n])*(d_Log[batch][out_c][x_m-half_kernel][y_m-half_kernel][z_m-half_kernel][n]);
    }

    t2 = sqrt(t2);
    scalar_t t3 = acos(inner);
    // Check if t2 == 0 before dividing by it. t2 == 0 => X = X_m => d_X = 0
    if(abs(t2) < EPS){
        return;
    }
    else{
        scalar_t t4 = t3/pow(t2, 3);
        scalar_t t6 = t3/t2;
        scalar_t weight = weights[out_c][i][j][k][c];

        for(int n = 0; n < PER_VOXEL; n++){
            // compute d_X:
            //t6 I
            atomicAdd(&d_X[batch][c][x][y][z][n], weight*d_Log[batch][out_c][x_m-half_kernel][y_m-half_kernel][z_m-half_kernel][n]*t6);
            // - (1-t0^2)^(-0.5)/t2*T5
            if(abs(t2*sqrt((1-pow(inner,2)))) > EPS)
                atomicAdd(&d_X[batch][c][x][y][z][n], -weight*1/(t2*sqrt((1-pow(inner,2))))*X[batch][mid_c][x_m][y_m][z_m][n]*inner_dLog_t1);

            // - t4*t1*Transpose[t1]
            atomicAdd(&d_X[batch][c][x][y][z][n], -weight*t4*(X[batch][c][x][y][z][n]-inner*X[batch][mid_c][x_m][y_m][z_m][n])*inner_dLog_t1);
            // + t4*Transpose[t1]*m*T5
            atomicAdd(&d_X[batch][c][x][y][z][n], weight*t4*inner_m_t1*inner_dLog_t1*X[batch][mid_c][x_m][y_m][z_m][n]);
            // - t6*m*Transpose[m]
            atomicAdd(&d_X[batch][c][x][y][z][n], -weight*t6*inner_dLog_m*X[batch][mid_c][x_m][y_m][z_m][n]);

            //compute d_M:
            // - t0 t6 I
            atomicAdd(&d_M[batch][mid_c][x_m][y_m][z_m][n], -weight*d_Log[batch][out_c][x_m-half_kernel][y_m-half_kernel][z_m-half_kernel][n]*inner*t6);
            // - (1-t0^2)^(-0.5)/t3*T5
            if(abs(t2*sqrt((1-pow(inner,2)))) > EPS)
                atomicAdd(&d_M[batch][mid_c][x_m][y_m][z_m][n], -weight*1/(t2*sqrt((1-pow(inner,2))))*X[batch][c][x][y][z][n]*inner_dLog_t1);

            // - t0*t4*t1*Transpose[t1]
            atomicAdd(&d_M[batch][mid_c][x_m][y_m][z_m][n], weight*inner*t4*(X[batch][c][x][y][z][n]-inner*X[batch][mid_c][x_m][y_m][z_m][n])*inner_dLog_t1);
            // + t4*Transpose[t1]*m*T5
            atomicAdd(&d_M[batch][mid_c][x_m][y_m][z_m][n], weight*t4*inner_m_t1*inner_dLog_t1*X[batch][c][x][y][z][n]);
            // - t6*m*Transpose[v]
            atomicAdd(&d_M[batch][mid_c][x_m][y_m][z_m][n], -weight*t6*inner_dLog_m*X[batch][c][x][y][z][n]);
        }
    }
}


template <typename scalar_t>
__global__ void mvc_cuda_log_backward_kernel(
            torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> X,
            const torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits, size_t> __restrict__ weights,
            torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> d_log,
            torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> d_X,
            dim3 block_grid_dims){
    // compute block coordinates from linearized blockIdx.x
    size_t batch = blockIdx.x;
    size_t block_offset = blockIdx.y;

    const dim3 blockCoord = offset2coordinates_RM_log_bw(block_offset, block_grid_dims.x, block_grid_dims.y, block_grid_dims.z);

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
    if(x < 0 || x - half_kernel_floor >= d_log.size(2) ||
       y < 0 || y - half_kernel_floor >= d_log.size(3) ||
       z < 0 || z - half_kernel_floor >= d_log.size(4))
        return;

    // iterate over channels and spatial dimensions
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

                        // write out weighted Log to output array
                        for(int c_out = 0; c_out < weights.size(0); c_out++){
                            d_mvcLog(X, d_log, d_X, d_X, weights,
                                    batch, c, mid_channel,
                                    x_o, y_o, z_o,
                                    x, y, z,
                                    half_kernel_floor, c_out,
                                    i+half_kernel-1, j+half_kernel-1, k+half_kernel-1);
                        }
                    }
                }
            }
        }
    }
}

torch::Tensor mvc_cuda_log_backward(torch::Tensor x,
                                    torch::Tensor weights,
                                    torch::Tensor d_log){
    // MVC Forward Pass:
    // x: [batches, channels_in, H, W, D, N]
    // m: [batches, channels_in, H_og, W_og, D_og, N]

    auto H_out = x.size(2);
    auto W_out = x.size(3);
    auto D_out = x.size(4);

    // allocate output of size [batches, channels_out, H_out, W_out, D_out, N]
    torch::Tensor d_X = torch::zeros({x.size(0), x.size(1), H_out, W_out, D_out, x.size(5)},
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

    AT_DISPATCH_FLOATING_TYPES(x.type(), "mvc_cuda_log_backward_kernel", ([&] {
                mvc_cuda_log_backward_kernel<scalar_t><<<blocks, threads>>>(
                    x.packed_accessor<scalar_t, 6, torch::RestrictPtrTraits, size_t>(),
                    weights.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits, size_t>(),
                    d_log.packed_accessor<scalar_t, 6, torch::RestrictPtrTraits, size_t>(),
                    d_X.packed_accessor<scalar_t, 6, torch::RestrictPtrTraits, size_t>(),
                    block_grid_dims);
    }));

    return d_X;
}

template <typename scalar_t, typename accessor>
__host__ __device__ void mvcLog_bw(
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
__global__ void mvc_cuda_log_backward_weights_kernel(
            torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> X,
            torch::PackedTensorAccessor<scalar_t, 6, torch::RestrictPtrTraits, size_t> d_Log,
            torch::PackedTensorAccessor<scalar_t, 5, torch::RestrictPtrTraits, size_t> d_weights,
            dim3 block_grid_dims){
    // compute block coordinates from linearized blockIdx.x
    size_t batch = blockIdx.x;
    size_t block_offset = blockIdx.y;

    const dim3 blockCoord = offset2coordinates_RM_log_bw(block_offset, block_grid_dims.x, block_grid_dims.y, block_grid_dims.z);

    const int kernel_size = d_weights.size(1);

    // kernel size must be odd
    const int half_kernel = (kernel_size+1)/2;
    const int half_kernel_floor = (kernel_size)/2;

    // compute output element coordinate that we are going to compute
    const size_t x = blockCoord.x*blockDim.x + threadIdx.x + half_kernel_floor;
    const size_t y = blockCoord.y*blockDim.y + threadIdx.y + half_kernel_floor;
    const size_t z = blockCoord.z*blockDim.z + threadIdx.z + half_kernel_floor;

    const size_t mid_channel = X.size(1)/2;



    // check we are within bounds
    if(x < 0 || x - half_kernel_floor >= d_Log.size(2) ||
       y < 0 || y - half_kernel_floor >= d_Log.size(3) ||
       z < 0 || z - half_kernel_floor >= d_Log.size(4))
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
                        mvcLog_bw(X, mvc_out,
                              batch, c, mid_channel,
                              x_o, y_o, z_o,
                              x, y, z);

                        // write out weighted Log to output array
                        for(int c_out = 0; c_out < d_weights.size(0); c_out++){
                            scalar_t grad = 0;
                            for(int n = 0; n < X.size(5); n++){
                                grad += mvc_out[n]*d_Log[batch][c_out][x-half_kernel_floor][y-half_kernel_floor][z-half_kernel_floor][n];
                            }

                            atomicAdd(&d_weights[c_out][i+half_kernel-1][j+half_kernel-1][k+half_kernel-1][c], grad);
                        }
                    }
                }
            }
        }
    }
}

torch::Tensor mvc_cuda_log_weights_backward(torch::Tensor x,
                                            torch::Tensor w,
                                            torch::Tensor d_Log){
    // MVC Forward Pass:
    // x: [batches, channels_in, H, W, D, N]
    // m: [batches, channels_in, H_og, W_og, D_og, N]

    auto H_out = w.size(1);
    auto W_out = w.size(2);
    auto D_out = w.size(3);

    // allocate output of size [batches, channels_out, H_out, W_out, D_out, N]
    torch::Tensor d_weights = torch::zeros({w.size(0), H_out, W_out, D_out, w.size(4)},
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

    AT_DISPATCH_FLOATING_TYPES(x.type(), "mvc_cuda_log_backward_weights_kernel", ([&] {
                mvc_cuda_log_backward_weights_kernel<scalar_t><<<blocks, threads>>>(
                    x.packed_accessor<scalar_t, 6, torch::RestrictPtrTraits, size_t>(),
                    d_Log.packed_accessor<scalar_t, 6, torch::RestrictPtrTraits, size_t>(),
                    d_weights.packed_accessor<scalar_t, 5, torch::RestrictPtrTraits, size_t>(),
                    block_grid_dims);
    }));

    return d_weights;
}

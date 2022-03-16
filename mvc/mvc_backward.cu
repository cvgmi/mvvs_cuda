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

template <typename scalar_t, typename accessor>
__device__ void d_mvcLog(
        accessor X,
        accessor d_Log,
        scalar_t* d_X,
        scalar_t* d_M,
        int batch, 
        int c, int mid_c,
        int x, int y, int z,
        int x_m, int y_m, int z_m){
    
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
        inner_dLog_t1 += (d_Log[batch][c][x][y][z][n])*(X[batch][c][x][y][z][n]-inner*X[batch][mid_c][x_m][y_m][z_m][n]);
    }

    // inner prodcut of middle point and t1
    scalar_t inner_m_t1 = 0;
    for(int n = 0; n < PER_VOXEL; n++){
        inner_m_t1 += (X[batch][mid_c][x_m][y_m][z_m][n])*(X[batch][c][x][y][z][n]-inner*X[batch][mid_c][x_m][y_m][z_m][n]);
    }

    // inner product of middle point and d_Log
    scalar_t inner_dLog_m = 0;
    for(int n = 0; n < PER_VOXEL; n++){
        inner_dLog_m += (X[batch][mid_c][x_m][y_m][z_m][n])*(d_Log[batch][c][x][y][z][n]);
    }

    t2 = sqrt(t2);
    scalar_t t3 = acos(inner);
    // Check if t2 == 0 before dividing by it. t2 == 0 => X = X_m => d_X = 0
    if(abs(t2) < EPS){
        for(int n = 0; n < PER_VOXEL; n++){
            d_X[batch][c][x][y][z][n] = 0;
            d_M[batch][c][x][y][z][n] = 0;
        }
    }
    else{
        scalar_t t4 = t3/pow(t2, 3);
        scalar_t t6 = t3/t2;

        for(int n = 0; n < PER_VOXEL; n++){
            // compute d_X:
            //t6 I
            d_X[batch][c][x][y][z][n] = d_Log[batch][c][x][y][z][n]*t6;
            // - (1-t0^2)^(-0.5)/t2*T5
            d_X[batch][c][x][y][z][n] -= 1/(t2*sqrt((1-pow(inner,2))))*X[batch][mid_c][x_m][y_m][z_m][n]*inner_dLog_t1;
            // - t4*t1*Transpose[t1]
            d_X[batch][c][x][y][z][n] -= t4*(X[batch][c][x][y][z][n]-inner*X[batch][mid_c][x_m][y_m][z_m][n])*inner_dLog_t1;
            // + t4*Transpose[t1]*m*T5
            d_X[batch][c][x][y][z][n] += t4*inner_m_t1*inner_dLog_t1*X[batch][mid_c][x_m][y_m][z_m][n];
            // - t6*m*Transpose[m]
            d_X[batch][c][x][y][z][n] -= t6*inner_dLog_m*X[batch][mid_c][x_m][y_m][z_m][n];

            //compute d_M:
            // - t0 t6 I
            d_M[batch][c][x][y][z][n] = -d_Log[batch][c][x][y][z][n]*inner*t6;
            // - (1-t0^2)^(-0.5)/t3*T5
            d_M[batch][c][x][y][z][n] -= 1/(t2*sqrt((1-pow(inner,2))))*X[batch][c][x][y][z][n]*inner_dLog_t1;
            // - t0*t4*t1*Transpose[t1]
            d_M[batch][c][x][y][z][n] += inner*t4*(X[batch][c][x][y][z][n]-inner*X[batch][mid_c][x_m][y_m][z_m][n])*inner_dLog_t1;
            // + t4*Transpose[t1]*m*T5
            d_M[batch][c][x][y][z][n] += t4*inner_m_t1*inner_dLog_t1*X[batch][c][x][y][z][n];
            // - t6*m*Transpose[v]
            d_M[batch][c][x][y][z][n] -= t6*inner_dLog_m*X[batch][c][x][y][z][n];
        }
    }
}

template <typename scalar_t, typename accessor>
__device__ void d_mvcExp(
        accessor X,
        accessor d_Exp,
        scalar_t* d_X,
        scalar_t* d_M,
        int batch, 
        int c, int mid_c,
        int x, int y, int z,
        int x_m, int y_m, int z_m){
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
        inner_dExp_m += X[batch][mid_c][x_m][y_m][z_m][n]*d_Exp[batch][c][x][y][z][n];


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
        d_M[batch][c][x][y][z][n] = d_Exp[batch][c][x][y][z][n]*cos_norm;
    }
}


// FOR TESTING:
template <typename scalar_t>
void d_mvcExp_host(
        torch::TensorAccessor<scalar_t, 6> X,
        torch::TensorAccessor<scalar_t, 6> d_Exp,
        torch::TensorAccessor<scalar_t, 6> d_X,
        torch::TensorAccessor<scalar_t, 6> d_M,
        int batch, 
        int c, int mid_c,
        int x, int y, int z,
        int x_m, int y_m, int z_m){
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
        inner_dExp_m += X[batch][mid_c][x_m][y_m][z_m][n]*d_Exp[batch][c][x][y][z][n];


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
        d_M[batch][c][x][y][z][n] = d_Exp[batch][c][x][y][z][n]*cos_norm;
    }
}

std::vector<torch::Tensor> exp_cuda_backward(torch::Tensor X, 
                                torch::Tensor d_Exp,
                                int batch, 
                                int c, int mid_c, 
                                int x, int y, int z,
                                int x_m, int y_m, int z_m){

    torch::Tensor d_X = X.clone()*0;
    torch::Tensor d_M = X.clone()*0;

    d_mvcExp_host(X.accessor<double, 6>(),
           d_Exp.accessor<double, 6>(),
           d_X.accessor<double, 6>(),
           d_M.accessor<double, 6>(),
           batch,
           c, mid_c,
           x, y, z,
           x_m, y_m, z_m);
    
    return {d_X, d_M};
}

template <typename scalar_t>
void d_mvcLog_host(
        torch::TensorAccessor<scalar_t, 6> X,
        torch::TensorAccessor<scalar_t, 6> d_Log,
        torch::TensorAccessor<scalar_t, 6> d_X,
        torch::TensorAccessor<scalar_t, 6> d_M,
        int batch, 
        int c, int mid_c,
        int x, int y, int z,
        int x_m, int y_m, int z_m){
    
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
        inner_dLog_t1 += (d_Log[batch][c][x][y][z][n])*(X[batch][c][x][y][z][n]-inner*X[batch][mid_c][x_m][y_m][z_m][n]);
    }

    // inner prodcut of middle point and t1
    scalar_t inner_m_t1 = 0;
    for(int n = 0; n < PER_VOXEL; n++){
        inner_m_t1 += (X[batch][mid_c][x_m][y_m][z_m][n])*(X[batch][c][x][y][z][n]-inner*X[batch][mid_c][x_m][y_m][z_m][n]);
    }

    // inner product of middle point and d_Log
    scalar_t inner_dLog_m = 0;
    for(int n = 0; n < PER_VOXEL; n++){
        inner_dLog_m += (X[batch][mid_c][x_m][y_m][z_m][n])*(d_Log[batch][c][x][y][z][n]);
    }

    t2 = sqrt(t2);
    scalar_t t3 = acos(inner);
    // Check if t2 == 0 before dividing by it. t2 == 0 => X = X_m => d_X = 0
    if(abs(t2) < EPS){
        for(int n = 0; n < PER_VOXEL; n++){
            d_X[batch][c][x][y][z][n] = 0;
            d_M[batch][c][x][y][z][n] = 0;
        }
    }
    else{
        scalar_t t4 = t3/pow(t2, 3);
        scalar_t t6 = t3/t2;

        for(int n = 0; n < PER_VOXEL; n++){
            // compute d_X:
            //t6 I
            d_X[batch][c][x][y][z][n] = d_Log[batch][c][x][y][z][n]*t6;
            // - (1-t0^2)^(-0.5)/t2*T5
            d_X[batch][c][x][y][z][n] -= 1/(t2*sqrt((1-pow(inner,2))))*X[batch][mid_c][x_m][y_m][z_m][n]*inner_dLog_t1;
            // - t4*t1*Transpose[t1]
            d_X[batch][c][x][y][z][n] -= t4*(X[batch][c][x][y][z][n]-inner*X[batch][mid_c][x_m][y_m][z_m][n])*inner_dLog_t1;
            // + t4*Transpose[t1]*m*T5
            d_X[batch][c][x][y][z][n] += t4*inner_m_t1*inner_dLog_t1*X[batch][mid_c][x_m][y_m][z_m][n];
            // - t6*m*Transpose[m]
            d_X[batch][c][x][y][z][n] -= t6*inner_dLog_m*X[batch][mid_c][x_m][y_m][z_m][n];

            //compute d_M:
            // - t0 t6 I
            d_M[batch][c][x][y][z][n] = -d_Log[batch][c][x][y][z][n]*inner*t6;
            // - (1-t0^2)^(-0.5)/t3*T5
            d_M[batch][c][x][y][z][n] -= 1/(t2*sqrt((1-pow(inner,2))))*X[batch][c][x][y][z][n]*inner_dLog_t1;
            // - t0*t4*t1*Transpose[t1]
            d_M[batch][c][x][y][z][n] += inner*t4*(X[batch][c][x][y][z][n]-inner*X[batch][mid_c][x_m][y_m][z_m][n])*inner_dLog_t1;
            // + t4*Transpose[t1]*m*T5
            d_M[batch][c][x][y][z][n] += t4*inner_m_t1*inner_dLog_t1*X[batch][c][x][y][z][n];
            // - t6*m*Transpose[v]
            d_M[batch][c][x][y][z][n] -= t6*inner_dLog_m*X[batch][c][x][y][z][n];
        }
    }
}

std::vector<torch::Tensor> log_cuda_backward(torch::Tensor X, 
                                torch::Tensor d_Log,
                                int batch, 
                                int c, int mid_c, 
                                int x, int y, int z,
                                int x_m, int y_m, int z_m){

    torch::Tensor d_X = X.clone()*0;
    torch::Tensor d_M = X.clone()*0;

    d_mvcLog_host(X.accessor<double, 6>(),
           d_Log.accessor<double, 6>(),
           d_X.accessor<double, 6>(),
           d_M.accessor<double, 6>(),
           batch,
           c, mid_c,
           x, y, z,
           x_m, y_m, z_m);
    
    return {d_X, d_M};
}
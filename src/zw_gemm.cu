//add by zw
// #include "zw.cuh"
extern "C"{
#include "zw_gemm.h"
}
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>

__global__ void gemm(float *A, float *B, float *C, int m, int n, int k, float alpha, float beta) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    // printf("col %d row %d\n", col, row);
    if( (col < n) && (row < m) )
    {
        float tmp = beta * C[row * n + col];
        for(int i = 0; i < k; ++i)
        {
            tmp += alpha * A[row * k + i] * B[col + i * n];
        }
        C[row * n + col] = tmp;
    }
}

__global__ void matrix_transpose(float *in, float *out, int rows, int cols){
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {
            out[j * rows + i] = in[i * cols + j];
    }
}
extern "C" void mysgemm(int TA, int TB, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){ 
    //M is row num of C matrix, while N is the col num of C, K is the col num of A 
    cudaDeviceSynchronize();
    // dim3 blockDim(1024);
    // dim3 gridDim(CEIL_DIV(M,32),CEIL_DIV(N,32));
    dim3 block(32, 8);
    dim3 grid( (N + block.x - 1) / block.x, (M + block.y - 1) / block.y );
    float *array1_t, *array2_t, *array3_t;
    cudaMalloc(&array1_t, sizeof(float)*K*M);
    cudaMalloc(&array2_t, sizeof(float)*N*K);
    cudaMalloc(&array3_t, sizeof(float)*N*M);
    if(TA==1){
        //transpose array1
        dim3 threadblock(16, 16);
        dim3 threadgrid(1 + K / threadblock.x, 1 + M / threadblock.y);
        matrix_transpose<<<threadgrid, threadblock>>>(A, array1_t, M, K);
    }
    if(TB==1){
        dim3 threadblock(16, 16);
        dim3 threadgrid(1 + N / threadblock.x, 1 + K / threadblock.y);
        matrix_transpose<<<threadgrid, threadblock>>>(B, array2_t, K, N);
    }
    int cases = TA*pow(2,1) + TB*pow(2,0);
    switch(cases){
        case 0: //no transpose
            // mysgemm_v4<<<gridDim, blockDim>>>(N, M, K, alpha, B, A, beta, C, lda, ldb, ldc);
            gemm <<<grid, block>>>(A, B, C, M, N, K, alpha, beta);
            break;
        case 1:
            gemm <<<grid, block>>>(A, array2_t, C, M, N, K, alpha, beta);
            break;
        case 2:
            gemm <<<grid, block>>>(array1_t, B, C, M, N, K, alpha, beta);
            break;
        case 3: //both are transposed
            // mysgemm_v4<<<gridDim, blockDim>>>(M, N, K, alpha, array1_t, array2_t, beta, C, lda, ldb, ldc);
            gemm <<<grid, block>>>(array2_t, array1_t, C, N, M, K, alpha, beta);
            // dim3 threadblock(16, 16);
            // dim3 threadgrid(1 + M / threadblock.x, 1 + N / threadblock.y);
            // matrix_transpose<<<threadgrid, threadblock>>>(C, array3_t, N, M);
            // cudaMemcpy(C, array3_t, sizeof(float)*N*M, cudaMemcpyDeviceToDevice);
            break;
    }
    cudaDeviceSynchronize();
    cudaFree(array1_t);
    cudaFree(array2_t);
    cudaFree(array3_t);
}



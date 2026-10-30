#include <cuda_runtime.h>

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int M, int N) {
    // Determine the global 2D thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure threads don't access out-of-bounds memory
    if (row < M && col < N) {
        // Convert 2D coordinates to a 1D index for row-major storage
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    // 16x16 threads per block
    dim3 threads(16, 16);
    // Calculate the number of blocks needed to cover the M x N matrix
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    
    // Launch the kernel
    matrix_add_kernel<<<blocks, threads>>>(A, B, C, M, N);
    
    // Wait for the GPU to finish
    cudaDeviceSynchronize();
}
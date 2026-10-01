#include <cuda_runtime.h>

__global__ void outer_product_kernel(const float* a, const float* b, float* C, int M, int N) {
    // Calculate global thread coordinates
    int j = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // Row index

    // Guard against out-of-bounds threads
    if (i < M && j < N) {
        // Compute the outer product and store in row-major order
        C[i * N + j] = a[i] * b[j];
    }
}

extern "C" void solve(const float* a, const float* b, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    outer_product_kernel<<<blocks, threads>>>(a, b, C, M, N);
    cudaDeviceSynchronize();
}

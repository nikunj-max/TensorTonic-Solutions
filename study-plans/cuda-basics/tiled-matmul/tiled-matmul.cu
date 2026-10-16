#include <cuda_runtime.h>

#define TILE_DIM 16

__global__ void tiled_matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Allocate shared memory for tiles of A and B
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    // Calculate global row and column indices
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;
    int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;

    // Loop over the tiles in the K dimension
    for (int t = 0; t < num_tiles; ++t) {
        // Cooperatively load tile of A into shared memory
        if (row < M && (t * TILE_DIM + threadIdx.x) < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_DIM + threadIdx.x];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Cooperatively load tile of B into shared memory
        if ((t * TILE_DIM + threadIdx.y) < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Wait for all threads to finish loading their elements
        __syncthreads();

        // Accumulate partial dot products from the shared memory tiles
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        // Wait for all threads to finish accumulating before the next tile load
        __syncthreads();
    }

    // Write the accumulated result to the global C matrix (check bounds)
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    tiled_matmul_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
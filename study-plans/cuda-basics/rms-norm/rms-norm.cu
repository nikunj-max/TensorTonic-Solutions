#include <cuda_runtime.h>
#include <math.h>

__global__ void rms_norm_kernel(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int row_offset = row * N;

    // Shared memory for block reduction
    __shared__ float s_sum[256]; // Matches the thread block size of 256

    // 1. Thread-local sum of squares
    // We stride through N in case N > blockDim.x
    float sum_sq = 0.0f;
    for (int j = tid; j < N; j += blockDim.x) {
        float val = input[row_offset + j];
        sum_sq += val * val;
    }
    s_sum[tid] = sum_sq;
    __syncthreads();

    // 2. Block-wide reduction to sum all squares
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    // 3. Compute inverse RMS 
    __shared__ float s_inv_rms;
    if (tid == 0) {
        float mean_sq = s_sum[0] / N;
        s_inv_rms = rsqrtf(mean_sq + eps); // Fast inverse square root instruction
    }
    __syncthreads();

    // 4. Normalize and apply learnable scale gamma
    float inv_rms = s_inv_rms;
    for (int j = tid; j < N; j += blockDim.x) {
        output[row_offset + j] = input[row_offset + j] * inv_rms * gamma[j];
    }
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M); // One block per row 
    rms_norm_kernel<<<blocks, threads>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}
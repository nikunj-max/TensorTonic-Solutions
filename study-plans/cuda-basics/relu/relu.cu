#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    // Calculate the global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure the thread is within the bounds of the array
    if (i < N) {
        // Apply ReLU: max(0, x) using CUDA's built-in fmaxf for single-precision floats
        output[i] = fmaxf(input[i], 0.0f);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
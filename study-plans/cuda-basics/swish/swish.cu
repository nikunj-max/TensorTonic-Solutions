#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* input, float* output, int N) {
    // Calculate the global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we don't access out of bounds
    if (idx < N) {
        // Read input once into a local variable (register) to avoid multiple global memory reads
        float x = input[idx];
        
        // Compute Swish: x / (1.0f + expf(-x))
        output[idx] = x / (1.0f + expf(-x));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    swish_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
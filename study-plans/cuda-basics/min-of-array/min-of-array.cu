#include <cuda_runtime.h>
#include <float.h>
#include <math.h>

// Helper to atomically find the minimum of floats using atomicCAS
__device__ void atomicMinFloat(float* address, float val) {
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int old = *address_as_ui, assumed;
    
    do {
        assumed = old;
        // If the existing value is already smaller or equal, we don't need to do anything
        if (__int_as_float(assumed) <= val) {
            break;
        }
        // Attempt to swap in the new minimum
        old = atomicCAS(address_as_ui, assumed, __float_as_int(val));
    } while (assumed != old);
}

__global__ void init_result(float* result) {
    result[0] = FLT_MAX;
}

__global__ void min_kernel(const float* input, float* result, int N) {
    // Statically allocated shared memory since threads=256 is fixed in the launcher
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory, or pad with FLT_MAX if out of bounds
    sdata[tid] = (i < N) ? input[i] : FLT_MAX;
    __syncthreads();

    // Perform a tree reduction within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    // Thread 0 of each block atomically updates the global minimum
    if (tid == 0) {
        atomicMinFloat(result, sdata[0]);
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Initialize the result buffer
    init_result<<<1, 1>>>(result);
    
    // Launch the kernel
    min_kernel<<<blocks, threads>>>(input, result, N);
    
    // Wait for GPU to finish before proceeding
    cudaDeviceSynchronize();
}
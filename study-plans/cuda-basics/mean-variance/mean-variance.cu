#include <cuda_runtime.h>

// Pass 1: One-pass reduction kernel to calculate sum and sum of squares
__global__ void mean_variance_kernel(const float* input, float* mean_out, float* var_out, int N) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Local thread accumulators
    float sum = 0.0f;
    float sq_sum = 0.0f;

    // Grid-stride loop processes multiple elements per thread if N > grid size
    for (int i = gid; i < N; i += stride) {
        float val = input[i];
        sum += val;
        sq_sum += val * val;
    }

    // Shared memory for block-level reduction
    __shared__ float s_sum[256];
    __shared__ float s_sq_sum[256];

    s_sum[tid] = sum;
    s_sq_sum[tid] = sq_sum;
    __syncthreads();

    // Tree reduction inside the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sq_sum[tid] += s_sq_sum[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 in each block adds its local sums to the global accumulators
    if (tid == 0) {
        atomicAdd(mean_out, s_sum[0]);
        atomicAdd(var_out, s_sq_sum[0]);
    }
}

// Pass 2: Tiny single-thread finalize kernel to calculate actual statistical values
__global__ void finalize_kernel(float* mean_out, float* var_out, int N) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float expected_x = mean_out[0] / N;
        float expected_x2 = var_out[0] / N;
        
        // Var(X) = E[X^2] - E[X]^2
        float var = expected_x2 - (expected_x * expected_x);
        
        // Prevent strictly negative zero variations due to floating point precision
        if (var < 0.0f) var = 0.0f;
        
        mean_out[0] = expected_x;
        var_out[0] = var;
    }
}

extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    // Cap blocks at 1024 to minimize atomic collisions and maximize grid-stride loop efficiency
    if (blocks > 1024) blocks = 1024;

    // Zero out device buffers safely (guarantees our atomic adds start from 0)
    cudaMemset(mean_out, 0, sizeof(float));
    cudaMemset(var_out, 0, sizeof(float));

    // Phase 1: Sum and Sum-Of-Squares
    mean_variance_kernel<<<blocks, threads>>>(input, mean_out, var_out, N);
    
    // Phase 2: Compute Mu and Sigma^2
    finalize_kernel<<<1, 1>>>(mean_out, var_out, N);

    // Sync before yielding to ensure buffers are updated and visible to the host
    cudaDeviceSynchronize();
}
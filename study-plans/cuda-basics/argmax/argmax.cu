#include <cuda_runtime.h>
#include <float.h>
#include <limits.h>

__global__ void argmax_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    __shared__ float s_vals[256];
    __shared__ int s_idxs[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    // Use -FLT_MAX and INT_MAX for out-of-bounds to ensure they lose comparisons
    if (idx < N) {
        s_vals[tid] = input[idx];
        s_idxs[tid] = idx;
    } else {
        s_vals[tid] = -FLT_MAX;
        s_idxs[tid] = INT_MAX;
    }
    __syncthreads();

    // Tree reduction inside the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float val1 = s_vals[tid];
            float val2 = s_vals[tid + s];
            int idx1 = s_idxs[tid];
            int idx2 = s_idxs[tid + s];

            // Keep the pair with the larger value. 
            // If values tie, keep the pair with the smaller index.
            if (val2 > val1 || (val2 == val1 && idx2 < idx1)) {
                s_vals[tid] = val2;
                s_idxs[tid] = idx2;
            }
        }
        __syncthreads();
    }

    // Thread 0 writes the block's winner to global scratch memory
    if (tid == 0) {
        block_vals[blockIdx.x] = s_vals[0];
        block_idxs[blockIdx.x] = s_idxs[0];
    }
}

__global__ void argmax_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    __shared__ float s_vals[256];
    __shared__ int s_idxs[256];

    int tid = threadIdx.x;
    
    // Each thread finds the max over its slice of blocks (needed if blocks > 256)
    float local_max_val = -FLT_MAX;
    int local_max_idx = INT_MAX;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float val = block_vals[i];
        int idx = block_idxs[i];
        if (val > local_max_val || (val == local_max_val && idx < local_max_idx)) {
            local_max_val = val;
            local_max_idx = idx;
        }
    }

    // Write thread-local winners to shared memory
    s_vals[tid] = local_max_val;
    s_idxs[tid] = local_max_idx;
    __syncthreads();

    // Tree reduction inside the single block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float val1 = s_vals[tid];
            float val2 = s_vals[tid + s];
            int idx1 = s_idxs[tid];
            int idx2 = s_idxs[tid + s];

            if (val2 > val1 || (val2 == val1 && idx2 < idx1)) {
                s_vals[tid] = val2;
                s_idxs[tid] = idx2;
            }
        }
        __syncthreads();
    }

    // Thread 0 writes the final overall winner to the result buffer
    if (tid == 0) {
        result[0] = s_idxs[0];
    }
}

extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals = nullptr;
    int* block_idxs = nullptr;
    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    argmax_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    argmax_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);
    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}
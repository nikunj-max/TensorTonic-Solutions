#include <cuda_runtime.h>
#include <math.h>

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int tid = threadIdx.x;
    int row = blockIdx.x;
    
    // Safety check just in case, though grid size M perfectly matches rows
    if (row >= M) return;

    __shared__ float s_sum[256];
    __shared__ float s_sum_sq[256];

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    int row_offset = row * N;

    // Step 1: Compute thread-local sum and sum of squares
    for (int i = tid; i < N; i += blockDim.x) {
        float val = input[row_offset + i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Step 2: Shared memory block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    // Step 3: Compute mean and inverse standard deviation 
    float mu = s_sum[0] / N;
    float var = s_sum_sq[0] / N - mu * mu;
    
    // Numerical safety (in rare float imprecisions where var can become slightly negative)
    if (var < 0.0f) var = 0.0f;
    
    // rsqrtf evaluates 1 / sqrtf(x) at hardware-level speed
    float inv_std = rsqrtf(var + eps);

    // Step 4: Normalize, apply gamma & beta, then write back to output
    for (int i = tid; i < N; i += blockDim.x) {
        float val = input[row_offset + i];
        float norm_val = (val - mu) * inv_std;
        output[row_offset + i] = norm_val * gamma[i] + beta[i];
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}

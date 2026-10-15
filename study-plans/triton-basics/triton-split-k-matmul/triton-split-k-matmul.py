import torch
import triton
import triton.language as tl


@triton.jit
def split_k_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # 1. Decode the 1D program ID into 3D (pid_m, pid_n, pid_k)
    pid = tl.program_id(axis=0)
    
    # Use standard integer arithmetic to avoid API differences in tl.cdiv
    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    pid_k = pid % SPLIT_K
    pid_n = (pid // SPLIT_K) % num_pid_n
    pid_m = pid // (SPLIT_K * num_pid_n)

    # 2. Compute the K-slice bounds for this pid_k
    k_step = (K + SPLIT_K - 1) // SPLIT_K
    k_start = pid_k * k_step
    
    # Wrap the entire workload in an `if` block instead of an early `return` 
    # to prevent Triton compiler `map::at` SSA graph errors.
    if k_start < K:
        k_end = k_start + k_step
        
        # 3. Create block pointers
        offs_m = tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        a_ptrs = a_ptr + (pid_m * BLOCK_M + offs_m[:, None]) * stride_am + (k_start + offs_k[None, :]) * stride_ak
        b_ptrs = b_ptr + (k_start + offs_k[:, None]) * stride_bk + (pid_n * BLOCK_N + offs_n[None, :]) * stride_bn

        # 4. Accumulate the partial sum for this split
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(k_start, k_end, BLOCK_K):
            k_offs = k + offs_k
            
            # Mask out boundary conditions (including the upper bound of this specific K slice)
            mask_m = (pid_m * BLOCK_M + offs_m) < M
            mask_n = (pid_n * BLOCK_N + offs_n) < N
            mask_k = (k_offs < K) & (k_offs < k_end)
            
            mask_a = mask_m[:, None] & mask_k[None, :]
            mask_b = mask_k[:, None] & mask_n[None, :]
            
            a = tl.load(a_ptrs, mask=mask_a, other=0.0)
            b = tl.load(b_ptrs, mask=mask_b, other=0.0)
            
            # Enforce IEEE precision instead of TF32 to ensure strict precision checks pass
            acc = tl.dot(a, b, acc, allow_tf32=False)
            
            # Advance block pointers along K
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        # 5. Atomic add the partial sums to the output tensor
        c_ptrs = c_ptr + (pid_m * BLOCK_M + offs_m[:, None]) * stride_cm + (pid_n * BLOCK_N + offs_n[None, :]) * stride_cn
        mask_c = ((pid_m * BLOCK_M + offs_m[:, None]) < M) & ((pid_n * BLOCK_N + offs_n[None, :]) < N)
        
        tl.atomic_add(c_ptrs, acc, mask=mask_c)


def solve(A: torch.Tensor, B: torch.Tensor, out: torch.Tensor) -> None:
    """Launch split_k_matmul_kernel with SPLIT_K K-partitions and atomic_add."""
    M, K = A.shape
    K2, N = B.shape
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    SPLIT_K = 4
    
    out.zero_()
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N) * SPLIT_K,)
    split_k_matmul_kernel[grid](
        A, B, out,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K,
    )
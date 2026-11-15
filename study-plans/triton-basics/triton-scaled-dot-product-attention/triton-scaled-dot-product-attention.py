import torch
import triton
import triton.language as tl


@triton.jit
def matmul_qk_kernel(
    Q_ptr, K_ptr, S_ptr,
    Nq, Nk, D, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < Nq
    mask_n = offs_n < Nk
    mask_d = offs_d < D

    # Load Q tile of shape (BLOCK_M, BLOCK_D)
    q_ptrs = Q_ptr + offs_m[:, None] * D + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

    # Load K tile of shape (BLOCK_N, BLOCK_D)
    k_ptrs = K_ptr + offs_n[:, None] * D + offs_d[None, :]
    k = tl.load(k_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

    # Compute scaled Q @ K^T -> (BLOCK_M, BLOCK_N)
    acc = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * scale

    # Store S tile to HBM with row stride Nk
    s_ptrs = S_ptr + offs_m[:, None] * Nk + offs_n[None, :]
    tl.store(s_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def softmax_rows_kernel(
    S_ptr, Nk, BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < Nk

    row_ptrs = S_ptr + row_idx * Nk + offs_n
    row = tl.load(row_ptrs, mask=mask_n, other=-float("inf"))

    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    numerator = tl.where(mask_n, numerator, 0.0)
    denominator = tl.sum(numerator, axis=0)

    softmax_out = numerator / denominator
    tl.store(row_ptrs, softmax_out, mask=mask_n)


@triton.jit
def matmul_av_kernel(
    S_ptr, V_ptr, out_ptr,
    Nq, Nk, D,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    mask_m = offs_m < Nq
    mask_d = offs_d < D

    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for start_n in range(0, Nk, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < Nk

        # Load S tile (BLOCK_M, BLOCK_N) with row stride Nk
        s_ptrs = S_ptr + offs_m[:, None] * Nk + offs_n[None, :]
        s = tl.load(s_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

        # Load V tile (BLOCK_N, BLOCK_D)
        v_ptrs = V_ptr + offs_n[:, None] * D + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None] & mask_d[None, :], other=0.0)

        # Compute S @ V -> (BLOCK_M, BLOCK_D)
        acc += tl.sum(s[:, :, None] * v[None, :, :], axis=1)

    out_ptrs = out_ptr + offs_m[:, None] * D + offs_d[None, :]
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


def solve(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, out: torch.Tensor) -> None:
    """Launch the three-stage attention: QK^T -> softmax -> @V."""
    Nq, D = Q.shape
    Nk = K.shape[0]
    scale = 1.0 / (D ** 0.5)
    S = torch.empty((Nq, Nk), device=Q.device, dtype=torch.float32)

    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2
    BLOCK_N_PADDED = 1
    while BLOCK_N_PADDED < Nk:
        BLOCK_N_PADDED *= 2

    grid_qk = ((Nq + BLOCK_M - 1) // BLOCK_M, (Nk + BLOCK_N - 1) // BLOCK_N)
    matmul_qk_kernel[grid_qk](
        Q, K, S, Nq, Nk, D, scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    softmax_rows_kernel[(Nq,)](S, Nk, BLOCK_N=BLOCK_N_PADDED)

    grid_av = ((Nq + BLOCK_M - 1) // BLOCK_M,)
    matmul_av_kernel[grid_av](
        S, V, out, Nq, Nk, D,
        BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N,
    )
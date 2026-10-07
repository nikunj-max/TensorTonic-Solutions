import torch
import triton
import triton.language as tl


@triton.jit
def layernorm_bwd_kernel(
    x_ptr, gamma_ptr, dy_ptr,
    dx_ptr, dgamma_ptr, dbeta_ptr,
    stride_x_row, stride_dy_row, stride_dx_row,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Set up pointers for the specific row
    x_row_ptr = x_ptr + pid * stride_x_row
    dy_row_ptr = dy_ptr + pid * stride_dy_row
    dx_row_ptr = dx_ptr + pid * stride_dx_row
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # 1. Load data
    x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)
    dy = tl.load(dy_row_ptr + cols, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + cols, mask=mask, other=0.0)
    
    # 2. Recompute forward statistics (mean, var, rstd, and normalized x)
    mean_x = tl.sum(x, axis=0) / N
    x_centered = tl.where(mask, x - mean_x, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    x_hat = x_centered * rstd
    
    # 3. Compute gradients intermediate constants (c1, c2)
    dy_norm = tl.where(mask, dy * gamma, 0.0)
    
    c1 = tl.sum(dy_norm, axis=0) / N
    c2 = tl.sum(dy_norm * x_hat, axis=0) / N
    
    # 4. Compute dx and store it directly per element
    dx = rstd * (dy_norm - c1 - x_hat * c2)
    tl.store(dx_row_ptr + cols, dx, mask=mask)
    
    # 5. Compute parameter gradients and accumulate globally via atomic adds
    dgamma_row = tl.where(mask, dy * x_hat, 0.0)
    dbeta_row = tl.where(mask, dy, 0.0)
    
    tl.atomic_add(dgamma_ptr + cols, dgamma_row, mask=mask)
    tl.atomic_add(dbeta_ptr + cols, dbeta_row, mask=mask)


def solve(
    x: torch.Tensor, gamma: torch.Tensor, dy: torch.Tensor,
    dx_out: torch.Tensor, dgamma_out: torch.Tensor, dbeta_out: torch.Tensor,
    eps: float,
) -> None:
    """Launch the LayerNorm backward kernel: one program per row, atomic reductions for dgamma and dbeta."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    
    # Pre-zero cross-row reduction buffers (dx_out is overwritten locally so it doesn't need zeroing)
    dgamma_out.zero_()
    dbeta_out.zero_()
    
    grid = (M,)
    layernorm_bwd_kernel[grid](
        x, gamma, dy,
        dx_out, dgamma_out, dbeta_out,
        x.stride(0), dy.stride(0), dx_out.stride(0),
        N, eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
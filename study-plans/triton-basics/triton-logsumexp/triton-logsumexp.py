import torch
import triton
import triton.language as tl


@triton.jit
def logsumexp_kernel(x_ptr, out_ptr, x_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # 1. Map program ID to row index
    row_idx = tl.program_id(0)
    
    # 2. Compute the start pointer for the current row
    row_start_ptr = x_ptr + row_idx * x_row_stride
    
    # 3. Create column offsets and mask for valid elements
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    
    # 4. Load the row
    # Crucial: Use -inf for masked out-of-bounds elements. 
    # This ensures they do not affect the row max, and because exp(-inf) = 0,
    # they contribute exactly 0 to the sum of exponentials.
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf'))
    
    # 5. Compute the maximum element in the row for numerical stability
    row_max = tl.max(row, axis=0)
    
    # 6. Stable log-sum-exp mathematical reduction
    # Subtract max, exponentiate, sum across the row, then log and add the max back
    shifted = row - row_max
    exp_vals = tl.exp(shifted)
    sum_exp = tl.sum(exp_vals, axis=0)
    lse = row_max + tl.log(sum_exp)
    
    # 7. Store the scalar result for this row
    # Output shape is (M,), so we just offset by row_idx
    tl.store(out_ptr + row_idx, lse)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch logsumexp_kernel with one program per row."""
    M, N = x.shape
    BLOCK_SIZE = triton.next_power_of_2(N)
    grid = (M,)
    logsumexp_kernel[grid](
        x, out, x.stride(0), N, BLOCK_SIZE=BLOCK_SIZE,
    )
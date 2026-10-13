import torch
import triton
import triton.language as tl

@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    
    # Calculate offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    # Load and mask out-of-bounds elements with 0.0
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Reduce this block to a single scalar sum
    block_sum = tl.sum(x, axis=0)
    
    # FIX: Add `+ 0` to the base pointer so Triton correctly resolves 
    # the address for the scalar atomic addition
    tl.atomic_add(out_ptr + 0, block_sum)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch sum_kernel on the provided tensors."""
    n = x.numel()
    out.zero_()
    
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    sum_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
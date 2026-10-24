import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Get the program ID for the 1D grid
    pid = tl.program_id(axis=0)
    
    # Calculate the memory offsets for this specific block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Mask out elements that are out of bounds (when N is not a multiple of BLOCK_SIZE)
    mask = offsets < n
    
    # Load data from memory into registers
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform element-wise addition
    result = x + y
    
    # Store the result back to memory
    tl.store(out_ptr + offsets, result, mask=mask)

def solve(x: torch.Tensor, y: torch.Tensor, out: torch.Tensor) -> None:
    """Launch vector_add_kernel on the provided tensors."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    vector_add_kernel[grid](x, y, out, n, BLOCK_SIZE=BLOCK_SIZE)
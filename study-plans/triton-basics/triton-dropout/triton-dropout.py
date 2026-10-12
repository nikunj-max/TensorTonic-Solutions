import torch
import triton
import triton.language as tl


@triton.jit
def dropout_kernel(
    x_ptr, mask_ptr, out_ptr,
    n, p,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Identify the current program (block) index
    pid = tl.program_id(axis=0)
    
    # 2. Compute the element offsets for this block
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 3. Create a mask to prevent out-of-bounds reads/writes at the tail
    mask_bounds = offs < n
    
    # 4. Load the input tensor and the 0/1 mask tensor
    x = tl.load(x_ptr + offs, mask=mask_bounds)
    m = tl.load(mask_ptr + offs, mask=mask_bounds)
    
    # 5. Compute the inverted scaling factor (broadcasts scalar to vector)
    scale = 1.0 / (1.0 - p)
    
    # 6. Apply elementwise dropout and scaling
    out = x * m * scale
    
    # 7. Store the result
    tl.store(out_ptr + offs, out, mask=mask_bounds)


def solve(x: torch.Tensor, mask: torch.Tensor, out: torch.Tensor, p: float) -> None:
    """Launch the dropout kernel: 1D grid over the input vector."""
    n = x.numel()
    BLOCK_SIZE = 1024
    
    # Launch one program per contiguous block of BLOCK_SIZE elements
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    dropout_kernel[grid](
        x, mask, out,
        n, p,
        BLOCK_SIZE=BLOCK_SIZE,
    )
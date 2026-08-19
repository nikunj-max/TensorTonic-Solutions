import numpy as np

def global_avg_pool(x):
    """
    Compute global average pooling over spatial dims.
    Supports (C,H,W) => (C,) and (N,C,H,W) => (N,C).
    """
    if x.ndim not in (3, 4):
        raise ValueError(f"Expected 3D or 4D tensor, got {x.ndim}D")
    
    # Average over the last two dimensions (H and W)
    return np.mean(x, axis=(-2, -1), dtype=np.float64)
import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Convert inputs to NumPy arrays to enable .ndim and vectorized math
    x = np.asarray(x, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    beta = np.asarray(beta, dtype=float)

    # Determine axes to reduce and reshape parameters based on input dimensions
    if x.ndim == 2:
        # (N, D): Normalize over the batch axis
        axis = 0
        gamma_broadcast = gamma
        beta_broadcast = beta
    elif x.ndim == 4:
        # (N, C, H, W): Normalize over batch, height, and width per channel
        axis = (0, 2, 3)
        # Reshape to (1, C, 1, 1) to broadcast correctly over (N, C, H, W)
        gamma_broadcast = gamma.reshape(1, -1, 1, 1)
        beta_broadcast = beta.reshape(1, -1, 1, 1)
    else:
        raise ValueError(f"Input x must be 2D or 4D, but got {x.ndim}D")

    # Compute mean and variance with keepdims=True for proper broadcasting
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)

    # Normalize, then scale and shift
    x_normalized = (x - mean) / np.sqrt(var + eps)
    out = gamma_broadcast * x_normalized + beta_broadcast

    return out
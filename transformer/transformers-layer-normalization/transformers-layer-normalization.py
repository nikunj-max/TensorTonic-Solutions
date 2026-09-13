import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns the last-axis normalized array.
    """
    # Ensure float64 computations as required
    x = x.astype(np.float64)
    gamma = gamma.astype(np.float64)
    beta = beta.astype(np.float64)
    
    # Calculate population mean and variance along the final axis
    # keepdims=True ensures the output retains the dimension for broadcasting
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    
    # Normalize the inputs
    x_normalized = (x - mu) / np.sqrt(var + eps)
    
    # Scale by gamma and shift by beta
    out = gamma * x_normalized + beta
    
    return out
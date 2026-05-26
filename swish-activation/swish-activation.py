import numpy as np

def swish(x):
    """
    Implement Swish activation function: x * sigmoid(x)
    """
    x = np.asarray(x, dtype=float)
    
    # Use np.clip to prevent overflow in exp(x)
    # For sigmoid(x) = 1 / (1 + exp(-x)), if x is very small (large negative), 
    # exp(-x) becomes very large.
    # Clipping ensures numerical stability while maintaining accuracy.
    x_clipped = np.clip(x, -500, 500)
    
    # Implement sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid = 1 / (1 + np.exp(-x_clipped))
    
    return x * sigmoid
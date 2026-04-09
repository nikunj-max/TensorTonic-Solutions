import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    # Ensure the input is a NumPy array to support vectorized operations
    x = np.asarray(x)
    
    # Use np.where to apply x if x >= 0, and alpha * x otherwise
    return np.where(x >= 0, x, alpha * x)
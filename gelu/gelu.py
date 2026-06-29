import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    # Convert input to a NumPy array of floats
    x_arr = np.asarray(x, dtype=float)
    
    # Vectorize the math.erf function so it can be applied element-wise to the array
    erf_vectorized = np.vectorize(math.erf)
    
    # Apply the exact GELU mathematical formula
    return 0.5 * x_arr * (1.0 + erf_vectorized(x_arr / np.sqrt(2.0)))

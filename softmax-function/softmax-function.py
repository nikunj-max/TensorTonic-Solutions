import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays or Python lists.
    """
    # Convert input to a numpy array to handle lists or tuples
    x = np.array(x)
    
    # Determine the axis: 1 for 2D arrays (rows), 0 for 1D arrays
    # Now x.ndim will work because x is a numpy array
    axis = 1 if x.ndim > 1 else 0
    
    # Subtract the maximum value for numerical stability
    x_max = np.max(x, axis=axis, keepdims=True)
    exponents = np.exp(x - x_max)
    
    # Compute the sum of exponents and divide
    sum_exponents = np.sum(exponents, axis=axis, keepdims=True)
    
    return exponents / sum_exponents
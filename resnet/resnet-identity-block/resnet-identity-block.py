import numpy as np

def identity_block(x, W1, W2):
    """
    Returns the identity residual-block output as a nested list.
    """
    # Convert inputs to NumPy float64 arrays
    x = np.array(x, dtype=np.float64)
    W1 = np.array(W1, dtype=np.float64)
    W2 = np.array(W2, dtype=np.float64)
    
    # Helper for ReLU activation
    relu = lambda v: np.maximum(0, v)
    
    # Main path
    F_x = relu(x @ W1.T)
    F_x = F_x @ W2.T
    
    # Add shortcut and apply final ReLU
    y = relu(F_x + x)
    
    # Round to 4 decimal places and convert to a nested list
    return np.round(y, 4).tolist()
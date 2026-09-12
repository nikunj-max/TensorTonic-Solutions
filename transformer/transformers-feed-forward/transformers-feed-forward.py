import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward output.
    """
    # Project to hidden space and apply bias
    hidden = x @ W1 + b1
    
    # Apply ReLU activation
    activated = np.maximum(0, hidden)
    
    # Project back to model dimension and apply output bias
    output = activated @ W2 + b2
    
    return output.astype(np.float64)
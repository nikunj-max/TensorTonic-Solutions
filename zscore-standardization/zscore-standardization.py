import numpy as np

def zscore_standardize(X, axis=0, eps=1e-12):
    """
    Standardize X: (X - mean)/std. If 2D and axis=0, per column.
    Return np.ndarray (float).
    """
    # Cast to float to avoid integer division issues and match required output type
    X = np.asarray(X, dtype=float)
    
    # Calculate mean and standard deviation along the specified axis
    mean = np.mean(X, axis=axis, keepdims=True)
    std = np.std(X, axis=axis, keepdims=True)
    
    # Standardize the features, adding eps to prevent division by zero
    return (X - mean) / (std + eps)
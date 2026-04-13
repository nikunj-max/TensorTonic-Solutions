import numpy as np

def covariance_matrix(X):
    """
    Compute covariance matrix from dataset X.
    """
    # Convert input to numpy array
    X = np.asarray(X, dtype=float)
    
    # Check if input is 2D and has at least 2 samples
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    
    n, d = X.shape
    
    # Step 1: Center the Data
    # Calculate the mean for each column (axis=0)
    mu = np.mean(X, axis=0)
    X_centered = X - mu
    
    # Step 2: Compute Covariance Matrix
    # Using the dot product of the transposed centered matrix and itself
    # Divide by (n - 1) for sample covariance
    covariance = np.dot(X_centered.T, X_centered) / (n - 1)
    
    return covariance
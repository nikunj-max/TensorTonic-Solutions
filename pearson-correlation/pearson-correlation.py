import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    X = np.array(X)
    
    # Check for invalid input: must be 2D and N >= 2
    if X.ndim != 2 or X.shape[0] < 2:
        return None
    
    N, D = X.shape
    
    # 1. Center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # 2. Compute the Covariance Matrix
    # Sigma = (X_centered.T @ X_centered) / (N - 1)
    covariance_matrix = np.dot(X_centered.T, X_centered) / (N - 1)
    
    # 3. Compute Standard Deviations
    # ddof=1 to match the N-1 used in covariance
    std_devs = np.std(X, axis=0, ddof=1)
    
    # 4. Compute the denominator matrix (sigma_i * sigma_j)
    denominator = np.outer(std_devs, std_devs)
    
    # 5. Compute Correlation Matrix: R = Sigma / denominator
    # We use np.divide and handle division by zero for zero variance features
    with np.errstate(divide='ignore', invalid='ignore'):
        correlation_matrix = covariance_matrix / denominator
        
    # 6. Ensure the diagonal is exactly 1.0 (unless variance is 0, then handled by NaN)
    # Hint 3: Handle zero variance features (already NaN from division), but keep diagonal 1.0
    for i in range(D):
        if std_devs[i] > 0:
            correlation_matrix[i, i] = 1.0
            
    return correlation_matrix
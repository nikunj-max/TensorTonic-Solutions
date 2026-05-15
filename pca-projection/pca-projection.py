import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Convert input list to a numpy array
    X = np.array(X)
    n, d = X.shape
    
    # 1. Center the data by subtracting the column means
    X_mean = np.mean(X, axis=0)
    X_c = X - X_mean
    
    # 2. Compute the sample covariance matrix (divide by n-1)
    # Using matrix multiplication: X_c^T @ X_c
    C = (X_c.T @ X_c) / (n - 1)
    
    # 3. Find eigenvalues and eigenvectors
    # np.linalg.eigh is optimized for symmetric matrices like covariance matrices
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    
    # eigh returns them in ascending order, so we need the indices to sort them descending
    sorted_indices = np.argsort(eigenvalues)[::-1]
    top_k_indices = sorted_indices[:k]
    
    # Extract the top-k eigenvectors (columns of the sorted eigenvector matrix)
    W = eigenvectors[:, top_k_indices]
    
    # 4. Project the centered data onto these k eigenvectors
    X_proj = X_c @ W
    
    # Return as an n x k list of floats
    return X_proj.tolist()
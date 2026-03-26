import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # 1. Ensure inputs are numpy arrays
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # 2. Handle 1D arrays by reshaping to 2D (n_samples, 1)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    # 3. Compute pairwise Euclidean distances using broadcasting
    # Shape: (n_test, 1, d) - (1, n_train, d) -> (n_test, n_train, d)
    diff = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    
    # Square differences, sum across dimensions, and take square root
    # Shape: (n_test, n_train)
    distances = np.sqrt(np.sum(diff**2, axis=2))

    # 4. Get indices of training points sorted by distance
    # argsort returns indices that would sort the array
    sorted_indices = np.argsort(distances, axis=1)

    # 5. Extract the k nearest neighbors
    # Handle the case where k might be larger than n_train
    n_train = X_train.shape[0]
    actual_k = min(k, n_train)
    
    k_neighbors = sorted_indices[:, :actual_k]

    # 6. Pad with -1 if k > n_train as per requirements
    if k > n_train:
        padding = np.full((X_test.shape[0], k - n_train), -1)
        k_neighbors = np.hstack([k_neighbors, padding])

    return k_neighbors.astype(int)
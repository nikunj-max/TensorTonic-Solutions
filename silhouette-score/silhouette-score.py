import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    n_samples = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        raise ValueError("Number of clusters must be at least 2.")
        
    # Step 1: Compute all-pairs Euclidean distances via broadcasting
    # (n_samples, 1, n_features) - (1, n_samples, n_features) -> (n_samples, n_samples)
    dists = np.sqrt(np.sum((X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2, axis=-1))
    
    # Create masks for every sample against every cluster
    # cluster_masks shape: (n_clusters, n_samples)
    cluster_masks = (labels == unique_labels[:, np.newaxis])
    
    # Pre-calculate count of samples in each cluster
    cluster_counts = cluster_masks.sum(axis=1)
    
    a = np.zeros(n_samples)
    b = np.full(n_samples, np.inf)
    
    # Step 2: Extract intra and inter-cluster averages
    for k_idx, label in enumerate(unique_labels):
        mask = cluster_masks[k_idx]
        count = cluster_counts[k_idx]
        
        if count > 1:
            # a(i): Mean distance to other points in the same cluster
            # sum over the row for the same cluster points, divided by (count - 1)
            a[mask] = np.sum(dists[mask][:, mask], axis=1) / (count - 1)
        else:
            a[mask] = 0.0
            
        # b(i): Mean distance to points in other clusters
        # Loop through all other clusters to find the minimum average distance
        other_masks = np.delete(cluster_masks, k_idx, axis=0)
        other_counts = np.delete(cluster_counts, k_idx)
        
        # Matrix multiply dists of current cluster points with masks of other clusters
        # to efficiently fetch sum of distances to other clusters
        inter_dist_sums = np.dot(dists[mask], other_masks.T)
        inter_dist_means = inter_dist_sums / other_counts
        
        # Find the minimum neighbor cluster mean distance for these points
        b[mask] = np.min(inter_dist_means, axis=1)

    # Step 3: Compute individual silhouette scores s(i)
    max_ab = np.maximum(a, b)
    
    # Handle edge case where max(a, b) == 0 (occurs if single-point clusters return 0)
    s = np.zeros(n_samples)
    valid = max_ab > 0
    s[valid] = (b[valid] - a[valid]) / max_ab[valid]
    
    return float(np.mean(s))
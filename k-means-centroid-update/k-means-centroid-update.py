def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    if not points:
        return []

    # Get the dimensionality of the points (e.g., 2 for 2D, 3 for 3D)
    dim = len(points[0])
    
    # Initialize sums for each cluster with zeros and a counter for points per cluster
    sums = [[0.0] * dim for _ in range(k)]
    counts = [0] * k

    # Accumulate the coordinates and counts for each assigned cluster
    for i in range(len(points)):
        cluster_idx = assignments[i]
        counts[cluster_idx] += 1
        for d in range(dim):
            sums[cluster_idx][d] += points[i][d]

    # Calculate the mean for each cluster
    new_centroids = []
    for j in range(k):
        if counts[j] == 0:
            # Requirement: If no points are assigned, return a zero vector
            new_centroids.append([0.0] * dim)
        else:
            # Divide the sum of each dimension by the number of points in the cluster
            mean_vector = [sums[j][d] / counts[j] for d in range(dim)]
            new_centroids.append(mean_vector)

    return new_centroids
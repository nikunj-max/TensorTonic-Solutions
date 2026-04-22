import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    Supports single point (3,) or batch (N, 3).
    """
    points = np.asanyarray(points)
    is_single_point = points.ndim == 1

    # Reshape single point (3,) to (1, 3) for consistent batch processing
    if is_single_point:
        points = points[np.newaxis, :]

    # Number of points in the batch
    num_points = points.shape[0]

    # Convert to homogeneous coordinates by appending a column of ones
    # points_h shape: (N, 4)
    ones = np.ones((num_points, 1), dtype=points.dtype)
    points_h = np.hstack([points, ones])

    # Apply transform: ph' = T @ ph.T
    # We transpose points_h to (4, N) so T (4, 4) can multiply it
    # Resulting shape: (4, N), then transpose back to (N, 4)
    transformed_h = (T @ points_h.T).T

    # Extract the spatial coordinates (first 3 columns)
    # Note: In standard rigid transforms, the last row of T is [0, 0, 0, 1],
    # so the 4th coordinate remains 1. 
    result = transformed_h[:, :3]

    # Return to original shape format
    if is_single_point:
        return result.flatten()
    
    return result
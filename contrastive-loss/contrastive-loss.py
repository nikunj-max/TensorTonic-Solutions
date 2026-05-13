import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    # Ensure inputs are numpy arrays for vectorized math
    a = np.asarray(a)
    b = np.asarray(b)
    y = np.asarray(y)

    # Compute Euclidean distance d_i = ||a_i - b_i||_2
    # axis=-1 handles both 1D (D,) and 2D (N, D) inputs
    distances = np.linalg.norm(a - b, axis=-1)

    # Similar pairs (y=1): minimize the distance squared
    similar_loss = y * np.square(distances)

    # Dissimilar pairs (y=0): maximize distance up to the margin
    # (1 - y) * max(0, m - d)^2
    dissimilar_loss = (1 - y) * np.square(np.maximum(0, margin - distances))

    # Sum the losses for each pair
    total_loss = similar_loss + dissimilar_loss

    # Apply the specified reduction
    if reduction == "mean":
        return float(np.mean(total_loss))
    elif reduction == "sum":
        return float(np.sum(total_loss))
    else:
        raise ValueError("reduction must be 'mean' or 'sum'")
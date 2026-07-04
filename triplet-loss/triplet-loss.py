import numpy as np

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss for embedding ranking.
    """
    # Convert inputs to arrays and ensure they are at least 2D (N, D)
    a = np.atleast_2d(anchor)
    p = np.atleast_2d(positive)
    n = np.atleast_2d(negative)
    
    # Compute squared Euclidean distance for positive and negative pairs
    d_ap = np.sum(np.square(a - p), axis=1)
    d_an = np.sum(np.square(a - n), axis=1)
    
    # Calculate the margin loss element-wise
    losses = np.maximum(0, d_ap - d_an + margin)
    
    # Return the scalar mean loss across the batch
    return np.mean(losses)
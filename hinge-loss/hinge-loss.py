import numpy as np

def hinge_loss(y_true, y_score, margin=1.0, reduction="mean") -> float:
    """
    Computes the Hinge Loss for binary SVM classification.
    
    y_true: 1D array of {-1, +1}
    y_score: 1D array of real scores, same shape as y_true
    margin: The margin parameter m (default 1.0)
    reduction: "mean" or "sum"
    Return: float
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Validate shapes
    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the same shape.")

    # Validate label set {-1, +1}
    if not np.all(np.isin(y_true, [-1, 1])):
        raise ValueError("y_true must only contain labels -1 or +1.")

    # Mathematical Definition: l_i = max(0, m - y_i * s_i)
    losses = np.maximum(0, margin - y_true * y_score)

    # Apply reduction
    if reduction == "mean":
        return float(np.mean(losses))
    elif reduction == "sum":
        return float(np.sum(losses))
    else:
        raise ValueError("Reduction must be 'mean' or 'sum'.")
import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through residual Jacobians.
    """
    result = np.asarray(x, dtype=np.float64).copy()
    for jacobian in gradients_F:
        jacobian = np.asarray(jacobian, dtype=np.float64)
        result = result + result @ jacobian
    return result

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Returns the gradient propagated through plain Jacobians.
    """
    result = np.asarray(x, dtype=np.float64).copy()
    for jacobian in gradients_F:
        jacobian = np.asarray(jacobian, dtype=np.float64)
        result = result @ jacobian
    return result
import numpy as np


def linear_regression_closed_form(X, y):
    """Compute the optimal weight vector using the normal equation."""
    # Convert inputs to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Compute the normal equation: w = (X^T * X)^(-1) * X^T * y
    w = np.linalg.inv(X.T @ X) @ X.T @ y

    # Return as a numpy array or a list of floats
    return w.tolist()
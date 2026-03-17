import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Convert inputs to numpy arrays for linear algebra operations
    X = np.array(X)
    y = np.array(y)
    
    # Get the number of features (columns)
    n_features = X.shape[1]
    
    # Compute X transpose
    XT = X.T
    
    # Create the identity matrix I of size (d x d)
    I = np.eye(n_features)
    
    # Calculate the core components: (XT * X + lambda * I)
    # This adds the L2 penalty to the diagonal of the covariance matrix
    regularized_matrix = XT @ X + lam * I
    
    # Compute the inverse of the regularized matrix
    # Note: Adding lambda * I makes this much more stable than OLS
    matrix_inverse = np.linalg.inv(regularized_matrix)
    
    # Multiply the inverse by XT and then by y: (inverse * XT) * y
    weights = matrix_inverse @ XT @ y
    
    # Return weights as a list of floats
    return weights.tolist()
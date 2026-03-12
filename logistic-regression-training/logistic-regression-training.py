import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    n_samples, n_features = X.shape
    
    # Initialize parameters
    w = np.zeros(n_features)
    b = 0.0
    
    for _ in range(steps):
        # 1. Linear combination
        model_output = np.dot(X, w) + b
        
        # 2. Apply sigmoid activation to get probabilities (p)
        p = _sigmoid(model_output)
        
        # 3. Calculate gradients
        # Error (p - y) shows the direction and magnitude of the miss
        error = p - y
        dw = (1 / n_samples) * np.dot(X.T, error)
        db = (1 / n_samples) * np.sum(error)
        
        # 4. Update parameters (Gradient Descent step)
        w -= lr * dw
        b -= lr * db
        
    return w, b
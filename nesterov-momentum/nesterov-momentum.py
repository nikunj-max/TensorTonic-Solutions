import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    # Convert inputs to NumPy arrays ensuring float type for operations
    w = np.array(w, dtype=float)
    v = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)
    
    # Step 2: Update Velocity using the provided formula
    # v = μ * v + η * g(w_look)
    new_v = momentum * v + lr * grad
    
    # Step 3: Update Weights by subtracting the new velocity
    # w = w - v
    new_w = w - new_v
    
    return new_w, new_v
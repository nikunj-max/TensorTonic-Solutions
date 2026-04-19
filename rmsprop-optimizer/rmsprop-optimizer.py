import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    
    Args:
        w: Current parameters (NumPy array)
        g: Gradients (NumPy array)
        s: Running squared gradient accumulator (NumPy array)
        lr: Learning rate (eta)
        beta: Decay factor
        eps: Stability constant (epsilon)
        
    Returns:
        tuple: (new_w, new_s) updated parameters and accumulator
    """
    # Convert inputs to numpy arrays to ensure vectorized operations
    w = np.array(w)
    g = np.array(g)
    s = np.array(s)

    # Step 1: Update Running Average of squared gradients
    # s_t = beta * s_{t-1} + (1 - beta) * g_t^2
    new_s = beta * s + (1 - beta) * (g ** 2)

    # Step 2: Parameter Update
    # w_t = w_{t-1} - (lr / (sqrt(new_s) + eps)) * g_t
    new_w = w - (lr / (np.sqrt(new_s) + eps)) * g

    return new_w, new_s
import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    """
    # Check if we need to return lists to match input type
    return_list = isinstance(w, list)
    
    # Convert inputs to NumPy arrays to handle vectorization over any shape
    w = np.array(w, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)
    
    # Step 1: Update First Moment (Exponential Moving Average of gradients)
    m_new = beta1 * m + (1.0 - beta1) * grad
    
    # Step 2: Update Second Moment (Exponential Moving Average of squared gradients)
    v_new = beta2 * v + (1.0 - beta2) * (grad ** 2)
    
    # Step 3: Nesterov-Adjusted Update
    # Look-ahead momentum using the freshly updated m_new
    m_hat = beta1 * m_new + (1.0 - beta1) * grad
    
    # Apply the parameter update (incorporating the np.sqrt for standard Adam/Nadam scaling)
    w_new = w - lr * (m_hat / (np.sqrt(v_new) + eps))
    
    # Match the output type to the input type
    if return_list:
        return w_new.tolist(), m_new.tolist(), v_new.tolist()
        
    return w_new, m_new, v_new
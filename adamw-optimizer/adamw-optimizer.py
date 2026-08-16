import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    w = np.array(w, dtype=float)
    m = np.array(m, dtype=float)
    v = np.array(v, dtype=float)
    grad = np.array(grad, dtype=float)
    
    # Step 1: Update First Moment
    m_t = beta1 * m + (1.0 - beta1) * grad
    
    # Step 2: Update Second Moment
    v_t = beta2 * v + (1.0 - beta2) * (grad ** 2)
    
    # Step 3: AdamW Parameter Update
    w_t = w - (lr * weight_decay * w) - (lr * m_t / (np.sqrt(v_t) + eps))
    
    return w_t, m_t, v_t
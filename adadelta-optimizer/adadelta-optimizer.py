import numpy as np

def adadelta_step(w: list, grad: list, E_grad_sq: list, E_update_sq: list, rho: float = 0.9, eps: float = 1e-6) -> dict:
    """
    Returns a dictionary with new_w, new_E_grad_sq, and new_E_update_sq.
    """
    # Convert inputs to NumPy arrays
    w_arr = np.array(w, dtype=float)
    grad_arr = np.array(grad, dtype=float)
    E_grad_sq_arr = np.array(E_grad_sq, dtype=float)
    E_update_sq_arr = np.array(E_update_sq, dtype=float)
    
    # 1. Update the running squared-gradient average
    new_E_grad_sq = rho * E_grad_sq_arr + (1 - rho) * (grad_arr ** 2)
    
    # 2. Compute the parameter change (RMS of updates / RMS of gradients * gradient)
    rms_update_prev = np.sqrt(E_update_sq_arr + eps)
    rms_grad_curr = np.sqrt(new_E_grad_sq + eps)
    delta_w = - (rms_update_prev / rms_grad_curr) * grad_arr
    
    # 3. Update the running squared-change average
    new_E_update_sq = rho * E_update_sq_arr + (1 - rho) * (delta_w ** 2)
    
    # 4. Update the parameters
    new_w = w_arr + delta_w
    
    return {
        "new_w": new_w,
        "new_E_grad_sq": new_E_grad_sq,
        "new_E_update_sq": new_E_update_sq
    }
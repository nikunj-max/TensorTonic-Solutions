import numpy as np

def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def _as2d(a, feat):
    """Convert 1D array to 2D and track if conversion happened"""
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a.reshape(1, feat), True
    return a, False

def gru_cell_forward(x, h_prev, params):
    """
    Implement the GRU forward pass for one time step.
    Supports shapes (D,) & (H,) or (N,D) & (N,H).
    """
    # Determine dimensions from weights
    D = params["Wz"].shape[0]
    H = params["Uz"].shape[0]
    
    # Ensure inputs are 2D for vectorized operations
    x_2d, x_was_1d = _as2d(x, D)
    h_2d, h_was_1d = _as2d(h_prev, H)
    
    # 1. Update Gate: z_t = sigmoid(x*Wz + h_{t-1}*Uz + bz)
    z_t = _sigmoid(np.dot(x_2d, params["Wz"]) + np.dot(h_2d, params["Uz"]) + params["bz"])
    
    # 2. Reset Gate: r_t = sigmoid(x*Wr + h_{t-1}*Ur + br)
    r_t = _sigmoid(np.dot(x_2d, params["Wr"]) + np.dot(h_2d, params["Ur"]) + params["br"])
    
    # 3. Candidate Hidden State: h_tilde = tanh(x*Wh + (r_t * h_{t-1})*Uh + bh)
    # Element-wise multiplication for reset gate
    reset_h = r_t * h_2d
    h_tilde = np.tanh(np.dot(x_2d, params["Wh"]) + np.dot(reset_h, params["Uh"]) + params["bh"])
    
    # 4. New Hidden State: h_t = (1 - z_t) * h_{t-1} + z_t * h_tilde
    h_t = (1 - z_t) * h_2d + z_t * h_tilde
    
    # If original inputs were 1D, squeeze output back to 1D
    if x_was_1d or h_was_1d:
        return h_t.flatten()
    
    return h_t
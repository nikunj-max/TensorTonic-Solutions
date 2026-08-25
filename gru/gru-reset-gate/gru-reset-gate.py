import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def reset_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_r: np.ndarray, b_r: np.ndarray) -> np.ndarray:
    """
    Compute reset gate: r_t = sigmoid(W_r @ [h, x] + b_r)
    """
    # Concatenate previous hidden state and current input along the last axis
    hx_concat = np.concatenate([h_prev, x_t], axis=-1)
    
    # Apply linear transformation: multiply by W_r transposed and add bias b_r
    # W_r has shape (H, H+D), so W_r.T has shape (H+D, H)
    z = hx_concat @ W_r.T + b_r
    
    # Apply sigmoid activation
    r_t = sigmoid(z)
    
    return r_t
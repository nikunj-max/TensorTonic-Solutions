import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def input_gate(h_prev: np.ndarray, x_t: np.ndarray,
               W_i: np.ndarray, b_i: np.ndarray,
               W_c: np.ndarray, b_c: np.ndarray) -> tuple:
    """Compute input gate and candidate memory."""
    # Concatenate previous hidden state and current input along the feature dimension
    concat = np.concatenate([h_prev, x_t], axis=-1)
    
    # Compute the input gate (sigmoid) which decides what to update
    i_t = sigmoid(concat @ W_i.T + b_i)
    
    # Compute the candidate memory (tanh) which proposes new candidate values
    c_tilde = np.tanh(concat @ W_c.T + b_c)
    
    return i_t, c_tilde
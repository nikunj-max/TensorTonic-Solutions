import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def output_gate(h_prev: np.ndarray, x_t: np.ndarray, C_t: np.ndarray,
                W_o: np.ndarray, b_o: np.ndarray) -> tuple:
    """Compute output gate and hidden state."""
    # Concatenate previous hidden state and current input along the feature dimension
    concat = np.concatenate([h_prev, x_t], axis=-1)
    
    # Compute the output gate activation
    o_t = sigmoid(concat @ W_o.T + b_o)
    
    # Compute the new hidden state using the updated cell state
    h_t = o_t * np.tanh(C_t)
    
    return o_t, h_t
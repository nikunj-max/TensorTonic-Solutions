import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t: np.ndarray, h_prev: np.ndarray, C_prev: np.ndarray,
              W_f: np.ndarray, W_i: np.ndarray, W_c: np.ndarray, W_o: np.ndarray,
              b_f: np.ndarray, b_i: np.ndarray, b_c: np.ndarray, b_o: np.ndarray) -> tuple:
    """Complete LSTM cell forward pass."""
    
    # Use axis=-1 to concatenate correctly whether inputs are 1D (H,) or 2D (N, H)
    concat_input = np.concatenate([h_prev, x_t], axis=-1)
    
    # Compute the forget gate f_t
    f_t = sigmoid(concat_input @ W_f.T + b_f)
    
    # Compute the input gate i_t and candidate C_tilde
    i_t = sigmoid(concat_input @ W_i.T + b_i)
    c_tilde = np.tanh(concat_input @ W_c.T + b_c)
    
    # Compute the output gate o_t
    o_t = sigmoid(concat_input @ W_o.T + b_o)
    
    # Update the cell state
    C_t = f_t * C_prev + i_t * c_tilde
    
    # Compute the new hidden state
    h_t = o_t * np.tanh(C_t)
    
    return (h_t, C_t)
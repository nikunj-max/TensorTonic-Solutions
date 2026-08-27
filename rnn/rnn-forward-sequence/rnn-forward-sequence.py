import numpy as np

def rnn_forward(X: np.ndarray, h_0: np.ndarray,
                W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray) -> tuple:
    """
    Forward pass through entire sequence.
    """
    batch, T, input_dim = X.shape
    h_curr = h_0
    h_list = []
    
    for t in range(T):
        # Extract input for the current time step (shape: batch, input_dim)
        x_t = X[:, t, :]
        
        # Compute the new hidden state
        h_curr = np.tanh(x_t @ W_xh.T + h_curr @ W_hh.T + b_h)
        
        # Store the intermediate hidden state
        h_list.append(h_curr)
        
    # Stack the hidden states along the time axis (axis 1)
    # Resulting shape: (batch, T, hidden_dim)
    hidden_states = np.stack(h_list, axis=1)
    
    # The final hidden state is the last computed state
    h_final = h_curr
    
    return hidden_states, h_final
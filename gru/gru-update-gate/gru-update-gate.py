import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def update_gate(h_prev: np.ndarray, x_t: np.ndarray,
                W_z: np.ndarray, b_z: np.ndarray) -> np.ndarray:
    """
    Compute update gate: z_t = sigmoid(W_z @ [h, x] + b_z)
    """
    # Concatenate previous hidden state and current input along the last axis
    concat = np.concatenate([h_prev, x_t], axis=-1)
    
    # Apply the weight matrix and bias
    # concat is (N, H+D) and W_z.T is (H+D, H), yielding (N, H)
    z_t_linear = concat @ W_z.T + b_z
    
    # Apply sigmoid activation function to squash values to [0, 1]
    z_t = sigmoid(z_t_linear)
    
    return z_t
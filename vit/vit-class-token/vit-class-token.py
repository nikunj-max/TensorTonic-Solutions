import numpy as np

def prepend_class_token(patches: np.ndarray,
                        cls_token: np.ndarray) -> np.ndarray:
    """
    Returns the float64 sequence with the class token at position zero.
    """
    # Extract dimensions
    B = patches.shape[0]
    D = patches.shape[2]
    
    # Broadcast the (1, 1, D) token to match the batch size: (B, 1, D)
    cls_token_broadcasted = np.broadcast_to(cls_token, (B, 1, D))
    
    # Concatenate the token and patches along the sequence dimension (axis=1)
    sequence = np.concatenate([cls_token_broadcasted, patches], axis=1)
    
    # Ensure the output is float64
    return sequence.astype(np.float64)
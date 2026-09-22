import numpy as np

def add_position_embedding(patches: np.ndarray,
                           pos_embed: np.ndarray) -> np.ndarray:
    """
    Returns the float64 tokens after adding position embeddings.
    
    Args:
        patches: NumPy array of shape (B, N, D) containing the input patches.
        pos_embed: NumPy array of shape (1, N, D) containing the position embeddings.
        
    Returns:
        NumPy array of shape (B, N, D) and dtype float64.
    """
    # NumPy natively broadcasts the size-1 batch dimension of pos_embed 
    # across the B dimension of patches.
    result = patches + pos_embed
    
    return result.astype(np.float64)
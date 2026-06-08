import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Create a copy to prevent modifying the input array in-place
    masked_scores = scores.copy()
    
    # Get the sequence length T from the last dimension
    T = scores.shape[-1]
    
    # Create a 2D boolean mask for the upper triangular part (above the main diagonal)
    # k=1 excludes the diagonal, keeping only future positions
    future_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    
    # Leverage NumPy broadcasting: the (T, T) mask automatically aligns 
    # with the last two dimensions of the (..., T, T) score tensor
    masked_scores[..., future_mask] = mask_value
    
    return masked_scores
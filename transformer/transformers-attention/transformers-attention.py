import torch
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Returns the scaled dot-product attention output.
    """
    # Hint 2: Use K.shape[-1] for the scale (d_k)
    d_k = K.shape[-1]
    
    # Hint 1: Transpose only the final two dimensions of K
    # K.transpose(-2, -1) swaps the sequence length and feature dimensions
    K_T = K.transpose(-2, -1)
    
    # Compute unnormalized attention scores and scale them by sqrt(d_k)
    scores = torch.matmul(Q, K_T) / math.sqrt(d_k)
    
    # Apply softmax along the last dimension (key-position axis)
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Hint 3: Multiply the attention weights by V
    output = torch.matmul(attention_weights, V)
    
    return output
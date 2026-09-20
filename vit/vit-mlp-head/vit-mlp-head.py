import numpy as np

def classification_head(encoder_output: np.ndarray,
                        W_head: np.ndarray) -> np.ndarray:
    """
    Returns float64 class logits with shape (B, C).
    """
    # 1. Select the [CLS] token at position zero for all batches
    # Shape: (B, D)
    cls_token = encoder_output[:, 0, :]
    
    # 2. Apply LayerNorm over the embedding coordinates (axis=-1)
    epsilon = 1e-6
    mean = np.mean(cls_token, axis=-1, keepdims=True)
    variance = np.var(cls_token, axis=-1, keepdims=True)
    
    # Normalize the token
    h_hat = (cls_token - mean) / np.sqrt(variance + epsilon)
    
    # 3. Multiply by the classification matrix (W_head)
    # Shape: (B, D) @ (D, C) -> (B, C)
    logits = h_hat @ W_head
    
    return logits.astype(np.float64)
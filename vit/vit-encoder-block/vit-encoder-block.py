import numpy as np

def vit_encoder_block(x: np.ndarray, num_heads: int,
                      Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray,
                      Wo: np.ndarray, W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """
    Returns the float64 output of one pre-normalized ViT encoder block.
    """
    B, N, D = x.shape
    H = num_heads
    dh = D // H

    # 1. Pre-LayerNorm for Multi-Head Self-Attention (MSA)
    mean_x = np.mean(x, axis=-1, keepdims=True)
    var_x = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean_x) / np.sqrt(var_x + 1e-6)

    # 2. Multi-Head Self-Attention (MSA)
    # Project inputs and reshape to (B, H, N, dh)
    Q = (x_norm @ Wq).reshape(B, N, H, dh).transpose(0, 2, 1, 3)
    K = (x_norm @ Wk).reshape(B, N, H, dh).transpose(0, 2, 1, 3)
    V = (x_norm @ Wv).reshape(B, N, H, dh).transpose(0, 2, 1, 3)

    # Calculate attention scores
    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(dh)
    
    # Softmax over the last dimension safely
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    A = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Apply attention to values and concatenate heads
    out_heads = A @ V
    out_concat = out_heads.transpose(0, 2, 1, 3).reshape(B, N, D)
    
    # Final MSA projection
    msa_out = out_concat @ Wo

    # 3. First Residual Connection
    y = x + msa_out

    # 4. Pre-LayerNorm for MLP
    mean_y = np.mean(y, axis=-1, keepdims=True)
    var_y = np.var(y, axis=-1, keepdims=True)
    y_norm = (y - mean_y) / np.sqrt(var_y + 1e-6)

    # 5. MLP
    # First layer linear projection
    hidden = y_norm @ W1
    
    # GELU activation (using the standard tanh approximation)
    gelu_hidden = 0.5 * hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (hidden + 0.044715 * (hidden ** 3))))
    
    # Second layer linear projection
    mlp_out = gelu_hidden @ W2

    # 6. Second Residual Connection
    z = y + mlp_out

    return z
import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Computes layer normalization over the last axis."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def multi_head_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray, 
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """Computes projected multi-head self-attention."""
    batch_size, seq_len, d_model = q.shape
    head_dim = d_model // num_heads
    
    # Linear projections
    Q = q @ W_q
    K = k @ W_k
    V = v @ W_v
    
    # Reshape and transpose for multi-head attention: (batch, num_heads, seq_len, head_dim)
    Q = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    
    # Scaled dot-product attention
    scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    
    # Softmax (numerically stable)
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attn_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
    
    # Apply attention weights to V
    attn_out = attn_weights @ V
    
    # Transpose back and concatenate heads: (batch, seq_len, d_model)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    # Final output projection
    return attn_out @ W_o

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, 
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Computes the position-wise feed-forward network."""
    return np.maximum(0, x @ W1 + b1) @ W2 + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                  W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray,
                  beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray,
                  num_heads: int) -> np.ndarray:
    """
    Returns the post-normalized Transformer encoder states.
    """
    # 1. Multi-Head Attention (MHA)
    mha_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    
    # 2. First Layer Normalization (post-MHA residual)
    z = layer_norm(x + mha_out, gamma1, beta1)
    
    # 3. Position-wise Feed-Forward Network (FFN)
    ffn_out = feed_forward(z, W1, b1, W2, b2)
    
    # 4. Second Layer Normalization (post-FFN residual)
    y = layer_norm(z + ffn_out, gamma2, beta2)
    
    return y.astype(np.float64)
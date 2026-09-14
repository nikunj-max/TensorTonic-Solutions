import numpy as np

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Returns projected multi-head attention outputs.
    """
    batch_size, query_len, d_model = Q.shape
    _, kv_len, _ = K.shape
    
    d_k = d_model // num_heads
    
    # 1. Project Q, K, and V
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v
    
    # 2. Reshape and transpose to separate heads
    # (batch, seq_len, num_heads, d_k) -> (batch, num_heads, seq_len, d_k)
    Q_split = Q_proj.reshape(batch_size, query_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_split = K_proj.reshape(batch_size, kv_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_split = V_proj.reshape(batch_size, kv_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # 3. Compute scaled dot-product attention scores
    # Transpose only the final two dimensions of K_split: (batch, num_heads, d_k, kv_len)
    scores = (Q_split @ K_split.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    
    # 4. Apply softmax over key positions (last axis)
    # Subtracted max for numerical stability
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # 5. Multiply by V
    head_outputs = attn_weights @ V_split
    
    # 6. Transpose back and concatenate heads
    # (batch, num_heads, query_len, d_k) -> (batch, query_len, num_heads, d_k) -> (batch, query_len, d_model)
    concat_outputs = head_outputs.transpose(0, 2, 1, 3).reshape(batch_size, query_len, d_model)
    
    # 7. Apply the final output projection matrix
    output = concat_outputs @ W_o
    
    return output.astype(np.float64)
import numpy as np

def layer_norm(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True, ddof=0)
    return (x - mean) / np.sqrt(var + 1e-6)

def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def multi_head_attention(x: np.ndarray, num_heads: int, Wq: np.ndarray, Wk: np.ndarray, Wv: np.ndarray, Wo: np.ndarray) -> np.ndarray:
    B, N, D = x.shape
    head_dim = D // num_heads

    Q = (x @ Wq).reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    K = (x @ Wk).reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)
    V = (x @ Wv).reshape(B, N, num_heads, head_dim).transpose(0, 2, 1, 3)

    attn_scores = (Q @ K.transpose(0, 1, 3, 2)) / np.sqrt(head_dim)
    
    # Softmax
    attn_scores_max = np.max(attn_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(attn_scores - attn_scores_max)
    attn_probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    attn_out = attn_probs @ V
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, N, D)
    
    return attn_out @ Wo

def vit_forward(image: np.ndarray, patch_size: int, num_heads: int,
                W_patch: np.ndarray, patch_bias: np.ndarray,
                cls_token: np.ndarray, pos_embed: np.ndarray,
                encoder_weights: list, W_head: np.ndarray) -> np.ndarray:
    B, H, W, C_in = image.shape
    
    # Extract complete patches
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    N = num_patches_h * num_patches_w
    
    image = image[:, :num_patches_h * patch_size, :num_patches_w * patch_size, :]
    
    # Reshape and flatten patches
    patches = image.reshape(B, num_patches_h, patch_size, num_patches_w, patch_size, C_in)
    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(B, N, patch_size * patch_size * C_in)
    
    # Linear Projection
    x = (patches @ W_patch) + patch_bias
    
    # Prepend CLS token
    cls_tokens = np.broadcast_to(cls_token, (B, 1, cls_token.shape[-1]))
    x = np.concatenate([cls_tokens, x], axis=1)
    
    # Add position embedding
    x = x + pos_embed
    
    # Transformer Encoder
    for weights in encoder_weights:
        # Attention block
        norm1 = layer_norm(x)
        attn_out = multi_head_attention(norm1, num_heads, weights["Wq"], weights["Wk"], weights["Wv"], weights["Wo"])
        x = x + attn_out
        
        # MLP block
        norm2 = layer_norm(x)
        mlp_hidden = gelu(norm2 @ weights["W1"])
        mlp_out = mlp_hidden @ weights["W2"]
        x = x + mlp_out
        
    # Classification Head
    cls_state = x[:, 0, :]
    cls_state = layer_norm(cls_state)
    logits = cls_state @ W_head
    
    return logits.astype(np.float64)
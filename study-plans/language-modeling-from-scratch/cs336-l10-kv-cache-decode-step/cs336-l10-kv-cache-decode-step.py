import torch
import math

def kv_cache_decode_step(
    query: torch.Tensor, new_key: torch.Tensor, new_value: torch.Tensor,
    key_cache: torch.Tensor, value_cache: torch.Tensor,
    num_query_heads: int, num_kv_heads: int,
) -> dict:
    """
    Returns a dict of tensors: new_key_cache, new_value_cache, output.
    """
    B, H_q, D = query.shape
    G = num_query_heads // num_kv_heads
    original_dtype = query.dtype
    
    # 1. Update Key and Value Caches
    # Reshape new_key and new_value from (B, H_kv, D) to (B, H_kv, 1, D)
    new_key_unsq = new_key.unsqueeze(2)
    new_value_unsq = new_value.unsqueeze(2)
    
    # Concatenate along the sequence dimension (dim=2) -> (B, H_kv, S+1, D)
    new_key_cache = torch.cat([key_cache, new_key_unsq], dim=2)
    new_value_cache = torch.cat([value_cache, new_value_unsq], dim=2)
    
    # 2. Reshape Query to group heads
    # From (B, H_q, D) -> (B, H_kv, G, D)
    q_reshaped = query.reshape(B, num_kv_heads, G, D)
    
    # 3. Perform attention arithmetic in float32 for stability
    q_f32 = q_reshaped.to(torch.float32)
    k_f32 = new_key_cache.to(torch.float32)
    v_f32 = new_value_cache.to(torch.float32)
    
    # Compute dot products: (B, H_kv, G, D) @ (B, H_kv, D, S+1) -> (B, H_kv, G, S+1)
    scores = torch.matmul(q_f32, k_f32.transpose(-2, -1)) / math.sqrt(D)
    
    # Softmax over the sequence dimension (dim=-1)
    attn_weights = torch.softmax(scores, dim=-1)
    
    # Weighted sum of values: (B, H_kv, G, S+1) @ (B, H_kv, S+1, D) -> (B, H_kv, G, D)
    out_f32 = torch.matmul(attn_weights, v_f32)
    
    # 4. Restore original dtype and flatten query heads back
    # From (B, H_kv, G, D) -> (B, H_q, D)
    output = out_f32.to(original_dtype).reshape(B, H_q, D)
    
    # Return exactly in the requested order
    return {
        "new_key_cache": new_key_cache,
        "new_value_cache": new_value_cache,
        "output": output
    }
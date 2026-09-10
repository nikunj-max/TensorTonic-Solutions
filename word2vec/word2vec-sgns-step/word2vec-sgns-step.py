import torch

def sgns_sgd_step(W_in: torch.Tensor, W_out: torch.Tensor,
                  center_id: int, pos_id: int,
                  neg_ids: torch.Tensor, lr: float) -> dict:
    """
    Returns updated W_in and W_out float64 tensors in a dictionary.
    """
    # Clone to avoid in-place modification and to keep untouched rows unchanged
    W_in_new = W_in.clone()
    W_out_new = W_out.clone()

    # Extract original embeddings for gradient computation
    v_c = W_in[center_id]
    u_o = W_out[pos_id]

    # Initialize gradients for the center vector and output rows
    # Dictionary accumulates gradients for output rows keyed by their token ID
    grad_W_out = {}
    
    # Positive sample
    s_o = torch.dot(v_c, u_o)
    sigma_o = torch.sigmoid(s_o)
    
    grad_v_c = (sigma_o - 1.0) * u_o
    grad_u_o = (sigma_o - 1.0) * v_c
    grad_W_out[pos_id] = grad_u_o.clone()

    # Negative samples
    for n_id_tensor in neg_ids:
        n_id = n_id_tensor.item()
        u_i = W_out[n_id]
        
        s_i = torch.dot(v_c, u_i)
        sigma_i = torch.sigmoid(s_i)
        
        grad_u_i = sigma_i * v_c
        grad_v_c += sigma_i * u_i
        
        # Accumulate in dictionary
        if n_id in grad_W_out:
            grad_W_out[n_id] += grad_u_i
        else:
            grad_W_out[n_id] = grad_u_i.clone()

    # Apply all accumulated gradients
    W_in_new[center_id] -= lr * grad_v_c
    for out_id, grad_out in grad_W_out.items():
        W_out_new[out_id] -= lr * grad_out

    return {"W_in": W_in_new, "W_out": W_out_new}
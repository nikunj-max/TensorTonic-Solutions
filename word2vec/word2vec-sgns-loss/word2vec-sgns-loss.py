import torch
import torch.nn.functional as F

def sgns_loss(center_vec: torch.Tensor, pos_vec: torch.Tensor,
              neg_vecs: torch.Tensor) -> torch.Tensor:
    """
    Returns the scalar float64 SGNS loss.
    """
    # Calculate the positive dot product
    pos_score = torch.dot(center_vec, pos_vec)
    
    # Calculate all negative dot products using matrix-vector multiplication
    neg_scores = torch.mv(neg_vecs, center_vec)
    
    # Compute the stable loss using softplus
    # -log(sigmoid(x)) is mathematically equivalent to softplus(-x)
    loss = F.softplus(-pos_score) + F.softplus(neg_scores).sum()
    
    return loss
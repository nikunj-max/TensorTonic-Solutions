import torch
import torch.nn.functional as F

def cbow_forward(context_ids: torch.Tensor, target_id: int,
                 W_in: torch.Tensor, W_out: torch.Tensor) -> torch.Tensor:
    """
    Returns the scalar float64 CBOW cross-entropy loss.
    """
    # 1. Look up the embeddings for the context words and average them
    h = W_in[context_ids].mean(dim=0)
    
    # 2. Compute the logits (scores) for the entire vocabulary
    logits = W_out @ h
    
    # 3. Compute numerically stable log-softmax over the vocabulary
    log_probs = F.log_softmax(logits, dim=0)
    
    # 4. Extract the negative log probability of the true target word
    loss = -log_probs[target_id]
    
    return loss
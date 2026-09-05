import torch

def subsample_keep_probs(counts: torch.Tensor,
                         t: float = 1e-5) -> torch.Tensor:
    """
    Returns the float64 keep probability for every vocabulary word.
    """
    # Convert counts to relative frequencies f(w)
    frequencies = counts / counts.sum()
    
    # Calculate the keep probability: sqrt(t / f(w))
    p_keep = torch.sqrt(t / frequencies)
    
    # Clamp probabilities to a maximum of 1.0 and ensure float64 type
    return torch.clamp(p_keep, max=1.0).to(torch.float64)
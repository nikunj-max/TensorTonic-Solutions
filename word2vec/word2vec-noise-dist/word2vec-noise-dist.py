import torch

def noise_distribution(counts: torch.Tensor,
                       alpha: float = 0.75) -> torch.Tensor:
    """
    Returns the float64 negative-sampling distribution over the vocabulary.
    """
    powered_counts = torch.pow(counts, alpha)
    return powered_counts / torch.sum(powered_counts)
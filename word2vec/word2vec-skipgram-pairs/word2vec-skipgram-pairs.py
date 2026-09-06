import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns the ordered center-context pairs as an int64 tensor.
    """
    tokens = token_ids.tolist()
    n = len(tokens)
    pairs = []
    
    for i in range(n):
        center_token = tokens[i]
        start = max(0, i - window)
        end = min(n, i + window + 1)
        
        for j in range(start, end):
            if i != j:
                pairs.append([center_token, tokens[j]])
                
    if not pairs:
        return torch.empty((0, 2), dtype=torch.int64)
        
    return torch.tensor(pairs, dtype=torch.int64)
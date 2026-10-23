import numpy as np

def temperature_data_mixture(token_counts: list, alpha: int | float, training_budget: int) -> dict:
    counts = np.array(token_counts, dtype=np.float64)
    N = len(counts)
    
    probabilities = np.zeros(N, dtype=np.float64)
    expected_tokens = np.zeros(N, dtype=np.float64)
    expected_epochs = np.zeros(N, dtype=np.float64)
    
    mask = counts > 0
    if not np.any(mask):
        return {
            "probabilities": probabilities,
            "expected_tokens": expected_tokens,
            "expected_epochs": expected_epochs
        }
    
    # Calculate logs only for positive sizes to avoid log(0) warnings
    log_counts = np.log(counts[mask])
    
    # Subtract the maximum log count *before* multiplying by alpha.
    # This ensures log_weights are <= 0. Exp(<=0) is in (0, 1], preventing overflow 
    # even when alpha is extremely large.
    max_log_count = np.max(log_counts)
    log_weights = alpha * (log_counts - max_log_count)
    
    weights = np.exp(log_weights)
    
    # Compute normalized probabilities
    probabilities[mask] = weights / np.sum(weights)
    
    # Compute expected tokens
    expected_tokens = training_budget * probabilities
    
    # Compute expected epochs securely using np.divide and its where argument (Hint 3)
    np.divide(
        expected_tokens, 
        counts, 
        out=expected_epochs, 
        where=mask
    )
    
    return {
        "probabilities": probabilities,
        "expected_tokens": expected_tokens,
        "expected_epochs": expected_epochs
    }
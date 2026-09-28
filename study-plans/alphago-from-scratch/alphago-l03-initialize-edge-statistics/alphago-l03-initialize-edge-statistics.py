import numpy as np

def initialize_mcts_edges(policy_logits: np.ndarray, legal_mask: np.ndarray) -> dict:
    """
    Returns: N as an int64 array; W, Q, and P as float64 arrays in a dictionary.
    """
    # Number of actions
    n_actions = len(policy_logits)
    
    # Initialize arrays with required dtypes
    N = np.zeros(n_actions, dtype=np.int64)
    W = np.zeros(n_actions, dtype=np.float64)
    Q = np.zeros(n_actions, dtype=np.float64)
    P = np.zeros(n_actions, dtype=np.float64)
    
    # Compute Softmax for legal actions
    legal_indices = np.flatnonzero(legal_mask)
    if legal_indices.size > 0:
        # Extract legal logits
        legal_logits = policy_logits[legal_indices]
        
        # Max-subtraction trick for numerical stability
        m = np.max(legal_logits)
        
        # Compute probabilities in float64
        exp_logits = np.exp(legal_logits - m, dtype=np.float64)
        legal_probs = exp_logits / np.sum(exp_logits)
        
        # Assign computed probabilities to the corresponding legal positions
        P[legal_indices] = legal_probs

    return {
        "N": N,
        "W": W,
        "Q": Q,
        "P": P
    }
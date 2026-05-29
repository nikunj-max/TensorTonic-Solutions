import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    """
    Compute InfoNCE Loss for contrastive learning.
    """
    # 1. Ensure inputs are numpy arrays
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)
    
    # 2. Compute similarity matrix by dot product and scale by temperature
    # S[i, j] represents the similarity between Z1[i] and Z2[j]
    S = np.dot(Z1, Z2.T) / temperature
    
    # 3. Apply numerical stability trick: subtract max row-wise
    # This prevents overflow when computing np.exp()
    S_max = np.max(S, axis=1, keepdims=True)
    S_stable = S - S_max
    
    # 4. Compute exp(S) for the denominator sum
    exp_S = np.exp(S_stable)
    sum_exp_S = np.sum(exp_S, axis=1)
    
    # 5. Extract the positive pairs (diagonal elements) from the stable matrix
    diag_S = np.diag(S_stable)
    
    # 6. Compute the log-softmax values for the positive pairs
    log_probs = diag_S - np.log(sum_exp_S)
    
    # 7. InfoNCE loss is the negative mean of these log probabilities
    loss = -np.mean(log_probs)
    
    return float(loss)
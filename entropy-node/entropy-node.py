import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Handle empty nodes: by convention, an empty set has 0 entropy
    if len(y) == 0:
        return 0.0
    
    # Get the counts for each unique class
    _, counts = np.unique(y, return_counts=True)
    
    # Calculate proportions (probabilities)
    probabilities = counts / len(y)
    
    # Compute entropy: -sum(p * log2(p))
    # We filter for p > 0 to maintain numerical stability and handle 0log(0) = 0
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return float(entropy)
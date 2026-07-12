import numpy as np

def cohens_kappa(rater1, rater2):
    """
    Compute Cohen's Kappa coefficient.
    """
    n = len(rater1)
    
    # Calculate observed agreement (p_o)
    p_o = sum(1 for a, b in zip(rater1, rater2) if a == b) / n
    
    # Get label frequencies to calculate expected agreement (p_e)
    labels1, counts1 = np.unique(rater1, return_counts=True)
    labels2, counts2 = np.unique(rater2, return_counts=True)
    
    # Map labels to their counts for constant time lookup
    freq1 = dict(zip(labels1, counts1))
    freq2 = dict(zip(labels2, counts2))
    
    all_labels = set(labels1).union(set(labels2))
    
    p_e = 0.0
    for k in all_labels:
        prob1 = freq1.get(k, 0) / n
        prob2 = freq2.get(k, 0) / n
        p_e += prob1 * prob2
        
    # Handle the degenerate case where denominator would be zero
    if p_e == 1.0:
        return 1.0
        
    # Calculate and return Cohen's Kappa
    return (p_o - p_e) / (1.0 - p_e)
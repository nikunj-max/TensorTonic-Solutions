import math

def ndcg(relevance_scores: list, k: int) -> float:
    """
    Returns NDCG as a float.
    """
    def compute_dcg(scores: list, k: int) -> float:
        dcg = 0.0
        for i, r in enumerate(scores[:k], start=1):
            dcg += (2**r - 1) / math.log2(i + 1)
        return dcg

    actual_dcg = compute_dcg(relevance_scores, k)
    
    # Ideal DCG requires sorting the relevance scores in descending order
    ideal_scores = sorted(relevance_scores, reverse=True)
    ideal_dcg = compute_dcg(ideal_scores, k)
    
    if ideal_dcg == 0:
        return 0.0
        
    return actual_dcg / ideal_dcg
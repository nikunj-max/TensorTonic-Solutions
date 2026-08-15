def hit_rate_at_k(recommendations, ground_truth, k):
    """
    Compute the hit rate at K.
    """
    if not recommendations:
        return 0.0
        
    hits = 0
    for recs, truths in zip(recommendations, ground_truth):
        # Slice the top-K recommendations and convert to a set
        top_k_set = set(recs[:k])
        # Convert ground truth items to a set
        truth_set = set(truths)
        
        # If the intersection is not empty, it's a hit for this user
        if top_k_set.intersection(truth_set):
            hits += 1
            
    return hits / len(recommendations)
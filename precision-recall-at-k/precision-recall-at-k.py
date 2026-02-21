def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Slice the recommended list to only consider the top-k items
    top_k = recommended[:k]
    
    # Convert relevant items to a set for O(1) membership lookups
    relevant_set = set(relevant)
    
    # Count how many of the top-k recommendations are actually relevant
    hits = sum(1 for item in top_k if item in relevant_set)
    
    # Precision@k: fraction of recommended items that are relevant
    precision = hits / k
    
    # Recall@k: fraction of all relevant items that were caught in top-k
    recall = hits / len(relevant)
    
    return [float(precision), float(recall)]
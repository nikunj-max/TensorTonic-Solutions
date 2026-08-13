def top_k_recommendations(scores, rated_indices, k):
    """
    Return indices of top-k unrated items by predicted score.
    """
    # Create a list of (score, index) pairs for items not in rated_indices
    unrated_items = [(scores[i], i) for i in range(len(scores)) if i not in rated_indices]
    
    # Sort the unrated items by score in descending order
    unrated_items.sort(key=lambda x: x[0], reverse=True)
    
    # Return the indices of the top k items
    return [item[1] for item in unrated_items[:k]]
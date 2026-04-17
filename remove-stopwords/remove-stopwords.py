def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Converting the stopwords list to a set provides O(1) average time complexity 
    # for lookups, making the overall filtering process significantly faster.
    stop_set = set(stopwords)
    
    # Use a list comprehension to build the new list while preserving order.
    return [token for token in tokens if token not in stop_set]
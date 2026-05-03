def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    word_counts = {}
    
    for sentence in sentences:
        for word in sentence:
            # If the word is already in the dictionary, increment its count.
            # Otherwise, initialize it with a count of 1.
            word_counts[word] = word_counts.get(word, 0) + 1
            
    return word_counts
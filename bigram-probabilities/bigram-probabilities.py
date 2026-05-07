def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    from collections import Counter

    # 1. Build vocabulary V from unique tokens
    vocab = sorted(list(set(tokens)))
    v_size = len(vocab)
    
    # 2. Count all bigrams (w1, w2)
    counts = Counter()
    # Also track total occurrences of each w1 as the start of a bigram
    # to simplify the denominator calculation
    context_totals = Counter()
    
    for i in range(len(tokens) - 1):
        w1, w2 = tokens[i], tokens[i+1]
        counts[(w1, w2)] += 1
        context_totals[w1] += 1
        
    # 3. Compute smoothed conditional probabilities for every pair in V x V
    # Formula: P(v|w) = (count(w, v) + 1) / (count(w) + |V|)
    probs = {}
    for w1 in vocab:
        # Denominator is the number of times w1 was followed by anything + |V|
        denominator = context_totals[w1] + v_size
        
        for w2 in vocab:
            numerator = counts[(w1, w2)] + 1
            probs[(w1, w2)] = numerator / denominator
            
    # Return as standard dicts for clean output
    return dict(counts), probs
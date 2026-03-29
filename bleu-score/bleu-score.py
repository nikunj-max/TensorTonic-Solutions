import math
from collections import Counter

def bleu_score(candidate, reference, max_n):
    """
    Compute the BLEU score for a candidate translation.
    """
    c_len = len(candidate)
    r_len = len(reference)
    
    if c_len == 0:
        return 0.0
    
    precisions = []
    
    for n in range(1, max_n + 1):
        # Extract n-grams for candidate
        cand_ngrams = [tuple(candidate[i:i+n]) for i in range(c_len - n + 1)]
        if not cand_ngrams:
            precisions.append(0.0)
            continue
            
        cand_counts = Counter(cand_ngrams)
        
        # Extract n-grams for reference
        ref_ngrams = [tuple(reference[i:i+n]) for i in range(r_len - n + 1)]
        ref_counts = Counter(ref_ngrams)
        
        # Calculate clipped counts
        clipped_hits = 0
        for ngram, count in cand_counts.items():
            clipped_hits += min(count, ref_counts.get(ngram, 0))
            
        precision = clipped_hits / len(cand_ngrams)
        precisions.append(precision)
    
    # If any precision is 0, the geometric mean (and thus BLEU) is 0
    if min(precisions) <= 0:
        return 0.0
    
    # Geometric mean of precisions: exp(1/N * sum(log(p_n)))
    # We use uniform weights (1/max_n) for each n-gram order
    avg_log_precision = sum(math.log(p) for p in precisions) / max_n
    geometric_mean = math.exp(avg_log_precision)
    
    # Calculation of Brevity Penalty (BP)
    if c_len > r_len:
        bp = 1.0
    else:
        bp = math.exp(1 - r_len / c_len)
        
    return bp * geometric_mean
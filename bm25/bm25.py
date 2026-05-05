import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    N = len(docs)
    if N == 0:
        return np.array([], dtype=float)

    # 1. Precompute document statistics
    doc_lengths = np.array([len(doc) for doc in docs])
    avgdl = np.mean(doc_lengths)
    
    # 2. Compute Document Frequency (df) for terms in the query
    # We only care about terms that are actually in our query
    unique_query_terms = set(query_tokens)
    doc_frequencies = Counter()
    
    # Pre-calculate term frequencies for each document to avoid re-counting later
    doc_term_counts = [Counter(doc) for doc in docs]
    
    for term in unique_query_terms:
        df_t = sum(1 for d_counts in doc_term_counts if term in d_counts)
        doc_frequencies[term] = df_t

    # 3. Compute IDF for each unique query term
    idfs = {}
    for term in unique_query_terms:
        df_t = doc_frequencies[term]
        # Using the provided formula: idf(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
        # This variant ensures IDF is always positive
        idf_val = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
        idfs[term] = idf_val

    # 4. Calculate BM25 scores
    scores = np.zeros(N, dtype=float)
    
    for i in range(N):
        doc_len = doc_lengths[i]
        counts = doc_term_counts[i]
        tmp_score = 0.0
        
        for term in query_tokens:
            if term not in idfs:
                continue
            
            tf = counts[term]
            idf = idfs[term]
            
            # BM25 Equation: 
            # score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
            
            tmp_score += idf * (numerator / denominator)
            
        scores[i] = tmp_score

    return scores
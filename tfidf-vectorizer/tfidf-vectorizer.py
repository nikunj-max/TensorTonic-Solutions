import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    if not documents:
        return np.array([]), []

    # 1. Tokenization and Vocabulary Building
    tokenized_docs = [doc.lower().split() for doc in documents]
    
    # Get unique words across all documents and sort them alphabetically
    vocab_set = set()
    for doc in tokenized_docs:
        vocab_set.update(doc)
    vocabulary = sorted(list(vocab_set))
    
    if not vocabulary:
        return np.zeros((len(documents), 0)), []

    # Map words to their indices for efficient matrix filling
    word_to_idx = {word: i for i, word in enumerate(vocabulary)}
    
    num_docs = len(documents)
    num_vocab = len(vocabulary)
    
    # 2. Calculate Document Frequency (df) for IDF
    # Count how many documents contain each word
    df = Counter()
    for doc_tokens in tokenized_docs:
        unique_tokens = set(doc_tokens)
        for token in unique_tokens:
            df[token] += 1
            
    # 3. Calculate IDF scores
    # idf(t) = log(N / df(t))
    idf = {}
    for word in vocabulary:
        idf[word] = math.log(num_docs / df[word])

    # 4. Build the TF-IDF Matrix
    tfidf_matrix = np.zeros((num_docs, num_vocab))
    
    for doc_idx, doc_tokens in enumerate(tokenized_docs):
        if not doc_tokens:
            continue
            
        doc_counts = Counter(doc_tokens)
        total_terms_in_doc = len(doc_tokens)
        
        for word, count in doc_counts.items():
            # tf(t, d) = count(t, d) / total_terms_in_d
            tf = count / total_terms_in_doc
            
            # tf-idf(t, d) = tf * idf
            col_idx = word_to_idx[word]
            tfidf_matrix[doc_idx, col_idx] = tf * idf[word]

    return tfidf_matrix, vocabulary
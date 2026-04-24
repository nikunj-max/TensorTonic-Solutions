import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Create a mapping of word -> index for O(1) average lookup time
    vocab_map = {word: i for i, word in enumerate(vocab)}
    
    # Initialize a 1D NumPy array of zeros with the size of the vocabulary
    bow_vector = np.zeros(len(vocab), dtype=int)
    
    # Iterate through each token in the sentence
    for token in tokens:
        # If the token exists in our fixed vocabulary, increment its count
        if token in vocab_map:
            index = vocab_map[token]
            bow_vector[index] += 1
            
    return bow_vector
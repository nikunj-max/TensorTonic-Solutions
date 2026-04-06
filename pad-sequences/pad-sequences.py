import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Handle the empty input case
    if not seqs:
        return np.zeros((0, 0 if max_len is None else max_len), dtype=int)

    # Determine the target length L
    if max_len is None:
        L = max(len(seq) for seq in seqs)
    else:
        L = max_len

    # Initialize the result array with the pad_value
    # Shape: (Number of sequences, Target length)
    N = len(seqs)
    result = np.full((N, L), pad_value, dtype=int)

    # Fill each row with the sequence data
    for i, seq in enumerate(seqs):
        if L == 0:
            continue
            
        # Determine how much of the sequence to copy (handle truncation)
        # We copy up to L elements from the sequence
        copy_len = min(len(seq), L)
        
        if copy_len > 0:
            result[i, :copy_len] = seq[:copy_len]

    return result
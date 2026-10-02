def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    """
    if vocab_size <= 256:
        return {"vocab": [], "merges": []}
    
    # Initialize the token_bytes map for the 256 base byte tokens
    token_bytes = {i: bytes([i]) for i in range(256)}
    
    # Convert corpus strings to lists of byte/token IDs
    corpus_ids = [list(s.encode('utf-8')) for s in corpus]
    
    vocab = []
    merges = []
    current_id = 256
    
    while current_id < vocab_size:
        counts = {}
        
        # Count overlapping adjacent pairs within each string sequence
        for seq in corpus_ids:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                counts[pair] = counts.get(pair, 0) + 1
                
        if not counts:
            # No more pairs to merge
            break
            
        # Find the most frequent pair.
        # Tie-breaker: lexicographically greater pair of byte strings
        best_pair = max(
            counts.keys(), 
            key=lambda p: (counts[p], (token_bytes[p[0]], token_bytes[p[1]]))
        )
        
        # Add to merges and vocab lists
        merges.append([best_pair[0], best_pair[1], current_id])
        
        new_token_bytes = token_bytes[best_pair[0]] + token_bytes[best_pair[1]]
        vocab.append([current_id, list(new_token_bytes)])
        
        # Update token_bytes map
        token_bytes[current_id] = new_token_bytes
        
        # Replace occurrences of the best pair without overlapping, from left to right
        for k in range(len(corpus_ids)):
            seq = corpus_ids[k]
            i = 0
            new_seq = []
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == best_pair[0] and seq[i+1] == best_pair[1]:
                    new_seq.append(current_id)
                    i += 2  # Skip the replaced pair
                else:
                    new_seq.append(seq[i])
                    i += 1
            corpus_ids[k] = new_seq
            
        current_id += 1
        
    return {"vocab": vocab, "merges": merges}
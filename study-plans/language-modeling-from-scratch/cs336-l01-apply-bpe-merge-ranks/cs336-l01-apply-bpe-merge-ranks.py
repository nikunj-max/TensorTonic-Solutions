def encode(text: str, merges: list[list[int]]) -> list[int]:
    """
    Returns token IDs after applying the ordered merge rules.
    """
    # Start with the UTF-8 bytes of the text as initial token IDs
    tokens = list(text.encode('utf-8'))
    
    # Apply each merge rule in the given order
    for l, r, n in merges:
        new_tokens = []
        i = 0
        n_len = len(tokens)
        
        # Scan through the tokens to find and replace non-overlapping pairs
        while i < n_len:
            if i < n_len - 1 and tokens[i] == l and tokens[i+1] == r:
                new_tokens.append(n)
                i += 2  # Skip the next token since it was merged
            else:
                new_tokens.append(tokens[i])
                i += 1
                
        tokens = new_tokens
        
    return tokens

def decode(ids: list[int], vocab: dict[int, list[int]]) -> str:
    """
    Returns the text reconstructed from the token bytes.
    """
    byte_seq = bytearray()
    
    # Concatenate the byte lists from each token
    for token_id in ids:
        byte_seq.extend(vocab[token_id])
        
    # Decode the complete byte sequence back into a UTF-8 string
    return byte_seq.decode('utf-8')
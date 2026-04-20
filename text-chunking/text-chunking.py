def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    chunks = []
    # Calculate the distance between the start of one chunk and the next
    step = chunk_size - overlap
    
    # Iterate through the tokens starting from 0, moving by the step size
    for i in range(0, len(tokens), step):
        # Slice the tokens from the current index to the chunk size
        chunk = tokens[i : i + chunk_size]
        chunks.append(chunk)
        
        # Stop if the current chunk reached or surpassed the end of the token list
        if i + chunk_size >= len(tokens):
            break
            
    return chunks
import math

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Compute dot product
    dot_product = sum(a * b for a, b in zip(x1, x2))
    
    # Compute magnitudes (norms) of both vectors
    norm_x1 = math.sqrt(sum(a * a for a in x1))
    norm_x2 = math.sqrt(sum(b * b for b in x2))
    
    # Calculate cosine similarity
    cosine_similarity = dot_product / (norm_x1 * norm_x2)
    
    # Calculate loss based on the label
    if label == 1:
        return float(1 - cosine_similarity)
    elif label == -1:
        return float(max(0.0, cosine_similarity - margin))
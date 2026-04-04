import math

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    n = len(actual_tokens)
    sum_log_prob = 0.0

    for i in range(n):
        # Extract the probability assigned to the actual token at position i
        p_i = prob_distributions[i][actual_tokens[i]]
        # Sum the natural logs of these probabilities
        sum_log_prob += math.log(p_i)

    # Compute cross-entropy (H): the negative average of the log-probabilities
    cross_entropy = -(1 / n) * sum_log_prob

    # Perplexity is the exponential of the cross-entropy
    return math.exp(cross_entropy)
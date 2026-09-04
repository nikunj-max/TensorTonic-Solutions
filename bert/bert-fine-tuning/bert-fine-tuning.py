import numpy as np

def bert_fine_tuning_step(hidden_states: np.ndarray, labels: np.ndarray,
                          classifier_W: np.ndarray, classifier_b: np.ndarray,
                          learning_rate: float) -> dict:
    """
    Returns updated classifier parameters and the pre-update loss.
    """
    # 1. Extract the hidden state at sequence position zero (CLS token)
    cls_states = hidden_states[:, 0, :]  # Shape: (B, D)
    batch_size = cls_states.shape[0]
    
    # 2. Compute logits
    logits = np.dot(cls_states, classifier_W) + classifier_b  # Shape: (B, C)
    
    # 3. Compute stable softmax probabilities
    shifted_logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted_logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # Shape: (B, C)
    
    # 4. Compute mean cross-entropy loss
    correct_probs = probs[np.arange(batch_size), labels]
    loss = float(-np.mean(np.log(correct_probs)))
    
    # 5. Compute gradients for logits (softmax-cross-entropy gradient)
    grad_logits = probs.copy()
    grad_logits[np.arange(batch_size), labels] -= 1
    grad_logits /= batch_size
    
    # 6. Compute analytical gradients for weights and biases
    grad_W = np.dot(cls_states.T, grad_logits)  # Shape: (D, C)
    grad_b = np.sum(grad_logits, axis=0)        # Shape: (C,)
    
    # 7. Apply exactly one gradient-descent update
    new_classifier_W = classifier_W - learning_rate * grad_W
    new_classifier_b = classifier_b - learning_rate * grad_b
    
    # 8. Return dictionary with correctly typed outputs
    return {
        "new_classifier_W": new_classifier_W.astype(np.float64),
        "new_classifier_b": new_classifier_b.astype(np.float64),
        "loss": loss
    }
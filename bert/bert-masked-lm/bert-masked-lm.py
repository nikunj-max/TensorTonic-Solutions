import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    mask_positions: np.ndarray,
    replace_probs: np.ndarray,
    random_tokens: np.ndarray,
    mask_token_id: int = 103
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns: tuple of (np.ndarray masked_ids, np.ndarray labels) with masking applied
    """
    # Create copies and initialize arrays
    masked_ids = np.copy(token_ids)
    labels = np.full_like(token_ids, -100)
    
    # Set labels for positions that are masked (to the original token ID)
    labels[mask_positions] = token_ids[mask_positions]
    
    # Apply the 80% [MASK] rule
    mask_condition = mask_positions & (replace_probs < 0.8)
    masked_ids[mask_condition] = mask_token_id
    
    # Apply the 10% random token rule
    random_condition = mask_positions & (replace_probs >= 0.8) & (replace_probs < 0.9)
    masked_ids[random_condition] = random_tokens[random_condition]
    
    # The remaining 10% (replace_probs >= 0.9) keep their original token IDs,
    # which is already handled since masked_ids is initialized as a copy of token_ids.
    
    return masked_ids, labels

class MLMHead:
    """Masked LM prediction head."""
    
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Predict token logits: hidden_states @ W + b
        """
        return hidden_states @ self.W + self.b
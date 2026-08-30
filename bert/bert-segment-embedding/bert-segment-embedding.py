import numpy as np

class BertEmbeddings:
    """
    BERT Embeddings = Token + Position + Segment
    """
    
    def __init__(self, vocab_size: int, max_position: int, hidden_size: int):
        self.hidden_size = hidden_size
        
        # Token embeddings
        self.token_embeddings = np.random.randn(vocab_size, hidden_size) * 0.02
        
        # Position embeddings (learned, not sinusoidal)
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        
        # Segment embeddings (just 2 segments: A and B)
        self.segment_embeddings = np.random.randn(2, hidden_size) * 0.02
    
    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        """
        Returns: np.ndarray of shape (batch, seq_len, hidden_size) with combined embeddings
        """
        batch_size, seq_len = token_ids.shape
        
        # Look up token embeddings: shape (batch_size, seq_len, hidden_size)
        tokens = self.token_embeddings[token_ids]
        
        # Create position indices [0, 1, ..., seq_len - 1] and look up embeddings: shape (seq_len, hidden_size)
        positions = np.arange(seq_len)
        positions_emb = self.position_embeddings[positions]
        
        # Look up segment embeddings: shape (batch_size, seq_len, hidden_size)
        segments = self.segment_embeddings[segment_ids]
        
        # Sum them all together (positions_emb will be broadcasted across the batch dimension)
        return tokens + positions_emb + segments

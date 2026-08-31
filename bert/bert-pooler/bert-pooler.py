import numpy as np

def tanh(x):
    return np.tanh(x)

class BertPooler:
    """
    BERT Pooler: Extracts [CLS] and applies dense + tanh.
    """
    
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.W = np.random.randn(hidden_size, hidden_size) * 0.02
        self.b = np.zeros(hidden_size)
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Returns: np.ndarray of shape (batch, hidden_size) with tanh-activated [CLS] output
        """
        # Extract the [CLS] token, which is at position 0 along the sequence dimension
        cls_hidden = hidden_states[:, 0, :]
        
        # Apply the linear projection (W * h + b) followed by tanh activation
        pooled_output = tanh(np.dot(cls_hidden, self.W) + self.b)
        
        return pooled_output

class SequenceClassifier:
    """
    Sequence classification head on top of BERT.
    """
    
    def __init__(self, hidden_size: int, num_classes: int):
        self.pooler = BertPooler(hidden_size)
        self.classifier = np.random.randn(hidden_size, num_classes) * 0.02
    
    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        """
        Returns: np.ndarray of shape (batch, num_classes) with classification logits
        """
        # Get the pooled [CLS] representation from the BertPooler
        pooled_output = self.pooler.forward(hidden_states)
        
        # Multiply by the classifier weight matrix to get the logits
        logits = np.dot(pooled_output, self.classifier)
        
        return logits

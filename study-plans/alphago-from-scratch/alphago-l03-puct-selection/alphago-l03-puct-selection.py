import numpy as np

def select_puct_action(visits: np.ndarray, total_values: np.ndarray, priors: np.ndarray, legal_mask: np.ndarray, cpuct: float) -> tuple[int, np.ndarray]:
    """
    Returns: a tuple of the Python action index and the float64 score array.
    """
    visits = visits.astype(np.float64)
    total_values = total_values.astype(np.float64)
    priors = priors.astype(np.float64)
    
    Q = np.divide(total_values, visits, out=np.zeros_like(total_values), where=visits > 0)
    T = np.sum(visits)
    
    U = cpuct * priors * (np.sqrt(T) / (1.0 + visits))
    
    scores = Q + U
    scores[~legal_mask] = -np.inf
    
    selected_action = int(np.argmax(scores))
    
    return selected_action, scores
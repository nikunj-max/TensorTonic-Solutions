import numpy as np

def td_value_update(V: list, s: int, r: float, s_next: int, alpha: float, gamma: float) -> np.ndarray:
    """
    Applies a single TD(0) update to state s and returns a new NumPy array.
    """
    # Create a copy of the input array to avoid modifying the original
    new_V = np.asarray(V, dtype=float).copy()
    
    # Compute the TD target and error
    td_target = r + gamma * new_V[s_next]
    td_error = td_target - new_V[s]
    
    # Update the value estimate for the current state
    new_V[s] = new_V[s] + alpha * td_error
    
    return new_V
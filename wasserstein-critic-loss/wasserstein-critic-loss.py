import numpy as np

def wasserstein_critic_loss(real_scores, fake_scores):
    """
    Compute Wasserstein Critic Loss for WGAN.
    
    Formula: L = E[D(fake)] - E[D(real)]
    
    Args:
        real_scores: np.ndarray - Critic outputs for real samples
        fake_scores: np.ndarray - Critic outputs for fake samples
        
    Returns:
        float: The calculated scalar loss.
    """
    # Calculate the mean of the critic's scores for fake and real samples
    mean_fake = np.mean(fake_scores)
    mean_real = np.mean(real_scores)
    
    # Return the difference as a scalar
    return float(mean_fake - mean_real)
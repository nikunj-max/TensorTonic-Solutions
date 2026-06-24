import math

def cosine_annealing_schedule(base_lr, min_lr, total_steps, current_step):
    """
    Compute the learning rate using cosine annealing.
    """
    # Calculate the cosine component: cos(pi * current_step / total_steps)
    cos_inner = math.pi * (current_step / total_steps)
    cos_component = math.cos(cos_inner)
    
    # Apply the cosine annealing formula
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos_component)
    
    return lr
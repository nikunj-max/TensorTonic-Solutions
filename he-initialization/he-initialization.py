import math

def he_initialization(W, fan_in):
    """
    Scale raw weights to He uniform initialization.
    """
    limit = math.sqrt(6 / fan_in)
    return [[w * 2 * limit - limit for w in row] for row in W]
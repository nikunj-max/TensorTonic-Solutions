import math

def cyclic_encoding(values, period):
    """
    Encode cyclic features as sin/cos pairs.
    """
    encoded_features = []
    for v in values:
        theta = 2 * math.pi * v / period
        encoded_features.append([math.sin(theta), math.cos(theta)])
        
    return encoded_features
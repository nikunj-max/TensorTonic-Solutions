import torch

def neuron_forward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor) -> tuple:
    """
    Returns a tuple of scalar tensors: preactivation and tanh output.
    """
    # Element-wise multiplication followed by sum handles the dot product, 
    # and safely returns 0 for empty vectors. Adding the bias gives the preactivation.
    a = torch.sum(inputs * weights) + bias
    
    # Apply the hyperbolic tangent activation function
    y = torch.tanh(a)
    
    return a, y

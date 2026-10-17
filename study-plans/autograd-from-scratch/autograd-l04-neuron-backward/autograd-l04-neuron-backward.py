import torch

def neuron_backward(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, upstream_gradient: torch.Tensor) -> tuple:
    """
    Returns a tuple of tensors: output, input gradients, weight gradients, bias gradient.
    """
    # 1. Forward Pass
    # Preactivation (a): sum of (inputs * weights) + bias
    a = torch.sum(inputs * weights) + bias
    
    # Output (y): tanh(a)
    y = torch.tanh(a)
    
    # 2. Backward Pass (Backpropagation)
    # Derivative of tanh(a) with respect to a is (1 - y^2)
    # Delta: g * (1 - y^2)
    delta = upstream_gradient * (1 - torch.square(y))
    
    # Gradient with respect to inputs: delta * w_i
    dx = delta * weights
    
    # Gradient with respect to weights: delta * x_i
    dw = delta * inputs
    
    # Gradient with respect to bias: delta
    db = delta
    
    return y, dx, dw, db
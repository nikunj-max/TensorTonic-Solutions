import torch

def scalar_neuron_module(inputs: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    """
    Returns a scalar tensor preserving the input dtype and device.
    """
    # Calculate the preactivation: a = b + sum(w_i * x_i)
    # torch.sum reduces the 1D element-wise product to a 0D scalar tensor.
    a = torch.sum(inputs * weights) + bias
    
    # Apply the tanh activation if non-linear path is selected
    if nonlinear:
        return torch.tanh(a)
    
    # Otherwise, return the exact linear preactivation
    return a
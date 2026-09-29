import torch

def dense_layer_forward(inputs: torch.Tensor, weight_matrix: torch.Tensor, biases: torch.Tensor, nonlinear: bool) -> torch.Tensor:
    """
    Returns an output vector in neuron order, preserving input dtype and device.
    """
    # Compute the matrix-vector product Wx and add biases b
    preactivations = torch.mv(weight_matrix, inputs) + biases
    
    # Apply the tanh activation function if nonlinear is True
    if nonlinear:
        return torch.tanh(preactivations)
    
    return preactivations
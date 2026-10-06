import torch

def train_tiny_micrograd_mlp(
    inputs: torch.Tensor, 
    targets: torch.Tensor, 
    weights: list[torch.Tensor], 
    biases: list[torch.Tensor], 
    learning_rate: float, 
    steps: int
) -> tuple:
    # Clone weights and biases to avoid modifying the supplied tensors
    W = [w.clone() for w in weights]
    b = [bias.clone() for bias in biases]
    
    loss_history = []
    
    # Reshape targets to (B, 1) for broadcasting with predictions
    targets_col = targets.view(-1, 1)
    
    for step in range(steps):
        # Forward pass
        H_list = [inputs]
        for i in range(len(W)):
            Z = torch.matmul(H_list[-1], W[i].T) + b[i]
            H = torch.tanh(Z)
            H_list.append(H)
        
        preds = H_list[-1]
        
        # Compute loss
        loss = torch.sum((preds - targets_col) ** 2)
        loss_history.append(loss)
        
        # Backward pass
        # Gradient of the loss with respect to the pre-activation of the final layer
        delta = 2 * (preds - targets_col) * (1 - preds ** 2)
        
        dW_list = []
        db_list = []
        
        # Backpropagate through the layers in reverse order
        for i in reversed(range(len(W))):
            dW = torch.matmul(delta.T, H_list[i])
            db = torch.sum(delta, dim=0)
            
            dW_list.insert(0, dW)
            db_list.insert(0, db)
            
            # Compute delta for the previous layer (if not the first layer)
            if i > 0:
                delta = torch.matmul(delta, W[i]) * (1 - H_list[i] ** 2)
                
        # Gradient descent update
        for i in range(len(W)):
            W[i] = W[i] - learning_rate * dW_list[i]
            b[i] = b[i] - learning_rate * db_list[i]
            
    # Final forward pass to compute final predictions and loss
    H = inputs
    for i in range(len(W)):
        Z = torch.matmul(H, W[i].T) + b[i]
        H = torch.tanh(Z)
        
    final_preds = H
    final_loss = torch.sum((final_preds - targets_col) ** 2)
    
    return final_preds.view(-1), final_loss, W, b, loss_history
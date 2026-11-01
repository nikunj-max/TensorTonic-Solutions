import torch

def policy_value_training_step(
    features: torch.Tensor,
    policy_weight: torch.Tensor,
    policy_bias: torch.Tensor,
    value_weight: torch.Tensor,
    value_bias: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    learning_rate: float,
    l2_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns detached float32 tensors: loss, policy weight/bias, value weight/bias.
    """
    policy_weight = policy_weight.detach().clone().requires_grad_(True)
    policy_bias = policy_bias.detach().clone().requires_grad_(True)
    value_weight = value_weight.detach().clone().requires_grad_(True)
    
    if value_bias.ndim == 0:
        value_bias = value_bias.detach().clone().unsqueeze(0).requires_grad_(True)
        is_scalar = True
    else:
        value_bias = value_bias.detach().clone().requires_grad_(True)
        is_scalar = False

    p_logits = torch.matmul(features, policy_weight.t()) + policy_bias
    log_p = torch.log_softmax(p_logits, dim=-1)
    policy_loss = -torch.mean(torch.sum(target_policies * log_p, dim=-1))

    v = torch.tanh(torch.matmul(features, value_weight) + value_bias.squeeze())
    value_loss = torch.mean((v - target_values) ** 2)

    l2_loss = l2_coefficient * (
        torch.sum(policy_weight ** 2) +
        torch.sum(policy_bias ** 2) +
        torch.sum(value_weight ** 2) +
        torch.sum(value_bias ** 2)
    )

    loss = policy_loss + value_loss + l2_loss

    grads = torch.autograd.grad(
        loss,
        (policy_weight, policy_bias, value_weight, value_bias),
    )

    new_pw = policy_weight - learning_rate * grads[0]
    new_pb = policy_bias - learning_rate * grads[1]
    new_vw = value_weight - learning_rate * grads[2]
    new_vb = value_bias - learning_rate * grads[3]

    if is_scalar:
        new_vb = new_vb.squeeze()

    return (
        loss.detach(),
        new_pw.detach(),
        new_pb.detach(),
        new_vw.detach(),
        new_vb.detach()
    )
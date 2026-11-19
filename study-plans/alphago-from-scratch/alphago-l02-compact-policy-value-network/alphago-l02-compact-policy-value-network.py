import torch
import torch.nn.functional as F


def compact_policy_value_network(
    x: torch.Tensor,
    trunk_weight: torch.Tensor,
    trunk_scale: torch.Tensor,
    trunk_shift: torch.Tensor,
    residual_parameters: list[dict],
    head_parameters: dict,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns: float32 tensors (policy_logits, values), shaped (B, N*N+1) and (B,).
    """
    device = x.device
    eps = float(epsilon)

    def to_f32(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=device, dtype=torch.float32)

    def channel_norm(
        z: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        mean = z.mean(dim=(0, 2, 3), keepdim=True)
        var = ((z - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        norm = (z - mean) / torch.sqrt(var + eps)
        return norm * to_f32(scale).view(1, -1, 1, 1) + to_f32(shift).view(1, -1, 1, 1)

    # Trunk: 3x3 padded conv -> BN -> ReLU
    h = F.conv2d(to_f32(x), to_f32(trunk_weight), bias=None, stride=1, padding=1)
    h = F.relu(channel_norm(h, trunk_scale, trunk_shift))

    # Residual tower
    for block in residual_parameters:
        r = F.conv2d(h, to_f32(block["conv1_weight"]), bias=None, stride=1, padding=1)
        r = F.relu(channel_norm(r, block["scale1"], block["shift1"]))
        r = F.conv2d(r, to_f32(block["conv2_weight"]), bias=None, stride=1, padding=1)
        r = channel_norm(r, block["scale2"], block["shift2"])
        h = F.relu(h + r)

    # Policy head: 1x1 conv -> BN -> ReLU -> flatten -> linear
    p = F.conv2d(
        h, to_f32(head_parameters["policy_conv_weight"]), bias=None, stride=1, padding=0
    )
    p = F.relu(
        channel_norm(p, head_parameters["policy_scale"], head_parameters["policy_shift"])
    )
    p = p.flatten(1)
    policy_logits = F.linear(
        p,
        to_f32(head_parameters["policy_linear_weight"]),
        to_f32(head_parameters["policy_linear_bias"]),
    )

    # Value head: 1x1 conv -> BN -> ReLU -> flatten -> linear -> ReLU -> linear -> tanh
    v = F.conv2d(
        h, to_f32(head_parameters["value_conv_weight"]), bias=None, stride=1, padding=0
    )
    v = F.relu(
        channel_norm(v, head_parameters["value_scale"], head_parameters["value_shift"])
    )
    v = v.flatten(1)
    v = F.relu(
        F.linear(
            v,
            to_f32(head_parameters["value_hidden_weight"]),
            to_f32(head_parameters["value_hidden_bias"]),
        )
    )
    values = torch.tanh(
        F.linear(
            v,
            to_f32(head_parameters["value_output_weight"]),
            to_f32(head_parameters["value_output_bias"]),
        )
    ).squeeze(-1)

    return policy_logits, values

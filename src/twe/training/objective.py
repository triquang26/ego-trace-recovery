import math

import torch
import torch.nn.functional as F
from torch import Tensor


def sample_flow_time(batch: int, endpoint_probability: float, device: torch.device) -> Tensor:
    s = torch.rand(batch, device=device)
    endpoint = torch.rand(batch, device=device) < endpoint_probability
    return torch.where(endpoint, torch.ones_like(s), s)


def noisy_controls(controls: Tensor, noise: Tensor, s: Tensor) -> Tensor:
    t = s.view(-1, 1, 1, 1)
    return (1 - t) * controls + t * noise


def masked_flow_loss(velocity: Tensor, target: Tensor, weight: Tensor) -> Tensor:
    error = (velocity.float() - target.float()).pow(2).sum(dim=(-1, -2))
    weight = weight.float()
    elements = velocity.shape[-1] * velocity.shape[-2]
    return (error * weight).sum() / (elements * weight.sum() + 1e-6)


def point_weights(mask: Tensor, moving: Tensor, static_weight: float) -> Tensor:
    return mask.float() * torch.where(moving, torch.ones_like(mask.float()), torch.full_like(mask.float(), static_weight))


def validity_loss(logits: Tensor, target: Tensor, anchor_mask: Tensor) -> Tensor:
    loss = F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")
    weight = anchor_mask.float()[..., None].expand_as(loss)
    return (loss * weight).sum() / (weight.sum() + 1e-6)


def warmup_cosine(step: int, total: int, warmup_fraction: float) -> float:
    warmup = max(1, int(total * warmup_fraction))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))

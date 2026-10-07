import math

import torch
from torch import Tensor, nn


def fourier_uv(uv: Tensor, frequencies: int) -> Tensor:
    scales = (2.0 ** torch.arange(frequencies, device=uv.device, dtype=uv.dtype)) * math.pi
    angles = (uv * 2 - 1)[..., None] * scales
    return torch.cat([uv * 2 - 1, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)


def sincos_2d(grid: int, dim: int) -> Tensor:
    quarter = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(quarter, dtype=torch.float64) / quarter))
    coords = torch.arange(grid, dtype=torch.float64)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    out_x = xx.flatten()[:, None] * omega
    out_y = yy.flatten()[:, None] * omega
    return torch.cat([out_x.sin(), out_x.cos(), out_y.sin(), out_y.cos()], dim=1).float()


def timestep_features(s: Tensor, dim: int) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=s.device, dtype=torch.float32) / half)
    angles = s.float()[:, None] * 1000.0 * freqs
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class TimeEmbedding(nn.Module):
    def __init__(self, feature_dim: int, width: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = nn.Sequential(nn.Linear(feature_dim, width), nn.SiLU(), nn.Linear(width, width))

    def forward(self, s: Tensor) -> Tensor:
        return self.mlp(timestep_features(s, self.feature_dim))

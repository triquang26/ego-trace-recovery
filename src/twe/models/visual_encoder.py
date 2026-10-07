import torch
from torch import Tensor, nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoVisualEncoder(nn.Module):
    def __init__(self, name: str, revision: str, grid: int):
        super().__init__()
        from transformers import Dinov2Model

        self.model = Dinov2Model.from_pretrained(name, revision=revision).eval().requires_grad_(False)
        self.grid = grid
        self.revision = f"{name}@{revision}"
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True) -> "DinoVisualEncoder":
        return super().train(False)

    @torch.no_grad()
    def forward(self, rgb: Tensor) -> Tensor:
        pixels = (rgb - self.mean) / self.std
        tokens = self.model(pixel_values=pixels).last_hidden_state[:, 1:]
        if tokens.shape[1] != self.grid * self.grid:
            raise ValueError(f"expected {self.grid ** 2} patch tokens, got {tokens.shape[1]}")
        return tokens.reshape(rgb.shape[0], self.grid, self.grid, -1)

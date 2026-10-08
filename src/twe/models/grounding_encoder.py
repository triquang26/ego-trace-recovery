import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from twe.models.visual_encoder import IMAGENET_MEAN, IMAGENET_STD


@dataclass
class GroundingOutput:
    text: Tensor
    text_mask: Tensor
    null: Tensor
    maps: list[Tensor]


def level_shapes(height: int, width: int, levels: int, stride: int = 8) -> list[tuple[int, int]]:
    shapes, h, w = [], math.ceil(height / stride), math.ceil(width / stride)
    for _ in range(levels):
        shapes.append((h, w))
        h, w = math.ceil(h / 2), math.ceil(w / 2)
    return shapes


def prompt(text: str | None) -> str:
    if not text:
        return "."
    text = text.strip().lower()
    return text if text.endswith(".") else text + "."


def split_levels(tokens: Tensor, shapes: list[tuple[int, int]]) -> list[Tensor]:
    maps, start = [], 0
    for h, w in shapes:
        maps.append(tokens[:, start : start + h * w].transpose(1, 2).reshape(tokens.shape[0], -1, h, w))
        start += h * w
    return maps


class GroundingDinoEncoder(nn.Module):
    def __init__(self, name: str, revision: str, max_length: int):
        super().__init__()
        from transformers import AutoProcessor, GroundingDinoForObjectDetection

        self.tokenizer = AutoProcessor.from_pretrained(name, revision=revision).tokenizer
        detector = GroundingDinoForObjectDetection.from_pretrained(name, revision=revision)
        self.model = detector.model.eval().requires_grad_(False)
        self.levels = detector.config.num_feature_levels
        self.max_length = max_length
        self.revision = f"{name}@{revision}"
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def train(self, mode: bool = True) -> "GroundingDinoEncoder":
        return super().train(False)

    @torch.no_grad()
    def forward(self, rgb: Tensor, image_valid: Tensor, instructions: list[str | None]) -> GroundingOutput:
        tokens = self.tokenizer([prompt(t) for t in instructions], padding="longest", return_tensors="pt")
        if tokens.input_ids.shape[1] > self.max_length:
            raise ValueError(f"instruction exceeds {self.max_length} tokens")
        tokens = {k: v.to(rgb.device) for k, v in tokens.items()}
        out = self.model(pixel_values=(rgb - self.mean) / self.std, pixel_mask=image_valid.long(), **tokens)
        shapes = level_shapes(rgb.shape[-2], rgb.shape[-1], self.levels)
        maps = split_levels(out.encoder_last_hidden_state_vision.float(), shapes)
        null = torch.tensor([not t for t in instructions], device=rgb.device)
        return GroundingOutput(out.encoder_last_hidden_state_text.float(), tokens["attention_mask"].bool(), null, maps)


def map_validity(image_valid: Tensor, shape: tuple[int, int]) -> Tensor:
    return F.adaptive_avg_pool2d(image_valid[:, None].float(), shape)[:, 0] > 0


def alignment_scores(local: Tensor, text: Tensor, text_mask: Tensor, null: Tensor) -> Tensor:
    scores = torch.einsum("bnc,blc->bnl", local, text) / math.sqrt(local.shape[-1])
    scores = scores.masked_fill(~text_mask[:, None], -1e4).amax(-1)
    return torch.where(null[:, None], torch.zeros_like(scores), scores)

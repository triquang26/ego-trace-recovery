import copy
from dataclasses import dataclass

from torch import Tensor, nn

from twe.models.embeddings import sincos_2d


@dataclass
class PromptFeatures:
    features: Tensor
    mask: Tensor
    self_mask: Tensor
    positions: Tensor


@dataclass
class FusedFeatures:
    visual: Tensor
    text: Tensor


class VisionProjection(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.input_norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, out_dim))
        self.skip = nn.Linear(in_dim, out_dim, bias=False)
        self.output_norm = nn.LayerNorm(out_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.output_norm(self.skip(tokens) + self.mlp(self.input_norm(tokens)))


class FusionLayer(nn.Module):
    def __init__(self, fusion: nn.Module, enhancer: nn.Module):
        super().__init__()
        self.fusion = fusion
        self.enhancer = enhancer

    def forward(self, visual: Tensor, visual_pad: Tensor, text: Tensor, prompt: PromptFeatures) -> tuple[Tensor, Tensor]:
        (visual, _), (text, _) = self.fusion(vision_features=visual, text_features=text,
                                             attention_mask_vision=visual_pad, attention_mask_text=~prompt.mask)
        text, _ = self.enhancer(hidden_states=text, attention_masks=prompt.self_mask,
                                position_embeddings=prompt.positions)
        return visual, text


def fusion_layers_from(encoder_layers, count: int) -> list[FusionLayer]:
    if count > len(encoder_layers):
        raise ValueError(f"requested {count} fusion layers, encoder has {len(encoder_layers)}")
    layers = []
    for layer in list(encoder_layers)[:count]:
        fusion = copy.deepcopy(layer.fusion_layer).requires_grad_(True)
        enhancer = copy.deepcopy(layer.text_enhancer_layer).requires_grad_(True)
        layers.append(FusionLayer(fusion, enhancer))
    return layers


class VisionLanguageFusion(nn.Module):
    def __init__(self, layers: list[FusionLayer], visual_dim: int, dim: int, grid: int, dropout: float):
        super().__init__()
        self.grid = grid
        self.projection = VisionProjection(visual_dim, dim, max(4 * dim, visual_dim // 2), dropout)
        self.register_buffer("position", sincos_2d(grid, dim), persistent=False)
        self.layers = nn.ModuleList(layers)

    def forward(self, patches: Tensor, patch_weight: Tensor, prompt: PromptFeatures) -> FusedFeatures:
        batch, grid, _, _ = patches.shape
        visual = self.projection(patches.flatten(1, 2)) + self.position
        visual_pad = patch_weight.flatten(1) <= 0
        text = prompt.features.masked_fill(~prompt.mask[..., None], 0.0)
        for layer in self.layers:
            visual, text = layer(visual, visual_pad, text, prompt)
        return FusedFeatures(visual.float().reshape(batch, grid, grid, -1), text.float())


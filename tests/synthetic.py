import hashlib
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from twe.config import Stage1Config, WorldConfig
from twe.models.fusion import PromptFeatures, VisionLanguageFusion, fusion_layers_from
from twe.models.grounding_encoder import GroundingOutput
from twe.models.world_module import WorldModule

TINY_WORLD = WorldConfig(width=64, layers=2, heads=4, ffn_width=128, time_embedding_dim=32, dropout=0.0)


class StubVisual(nn.Module):
    def __init__(self, dim: int = 768):
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(3, dim, 14, 14)
        self.revision = "stub-visual"

    def forward(self, rgb):
        return self.conv(rgb).permute(0, 2, 3, 1)


class StubGrounding(nn.Module):
    def __init__(self, dim: int = 256, length: int = 64):
        super().__init__()
        torch.manual_seed(1)
        self.embed = nn.Embedding(997, dim)
        self.conv = nn.Conv2d(3, dim, 8, 8)
        self.length = length
        self.revision = "stub-grounding"

    def forward(self, rgb, image_valid, instructions):
        device = self.embed.weight.device
        words = [(text or ".").split()[: self.length] for text in instructions]
        width = max(len(w) for w in words)
        ids = torch.zeros(len(words), width, dtype=torch.long, device=device)
        mask = torch.zeros(len(words), width, dtype=torch.bool, device=device)
        for row, sentence in enumerate(words):
            for col, word in enumerate(sentence):
                ids[row, col] = int(hashlib.md5(word.encode()).hexdigest(), 16) % 997
                mask[row, col] = True
        text = self.embed(ids) * mask[..., None]
        sentence = text.sum(1) / mask.sum(1, keepdim=True)
        level = self.conv(rgb) + sentence[..., None, None]
        maps = [level]
        for _ in range(3):
            maps.append(F.avg_pool2d(maps[-1], 2, ceil_mode=True))
        null = torch.tensor([not t for t in instructions], device=device)
        self_mask = (mask[:, :, None] & mask[:, None, :]) | torch.eye(width, dtype=torch.bool, device=device)
        positions = torch.zeros_like(text)
        return GroundingOutput(text, mask, null, maps, PromptFeatures(text, mask, self_mask, positions))


def grounding_layers(count: int):
    from transformers import GroundingDinoConfig
    from transformers.models.grounding_dino.modeling_grounding_dino import GroundingDinoEncoderLayer

    torch.manual_seed(2)
    config = GroundingDinoConfig(encoder_ffn_dim=256, fusion_dropout=0.0, fusion_droppath=0.0)
    layers = [GroundingDinoEncoderLayer(config).eval() for _ in range(count)]
    for layer in layers:
        layer.fusion_layer.vision_param.data.fill_(0.5)
        layer.fusion_layer.text_param.data.fill_(0.5)
    return layers


def tiny_module(cfg: WorldConfig = TINY_WORLD) -> WorldModule:
    fusion = None
    if cfg.fusion_layers:
        layers = fusion_layers_from(grounding_layers(cfg.fusion_layers), cfg.fusion_layers)
        fusion = VisionLanguageFusion(layers, cfg.visual_dim, cfg.text_dim, cfg.patch_grid, cfg.dropout)
    return WorldModule(cfg, StubVisual(), StubGrounding(), fusion)


def tiny_stage1(**overrides) -> Stage1Config:
    base = Stage1Config(world=TINY_WORLD, micro_batch_size=8, effective_batch_size=16, optimizer_updates=60,
                        learning_rate=1e-3, warmup_fraction=0.05, num_workers=0, log_every=10, eval_every=30,
                        eval_batches=2, checkpoint_every=30,
                        mixture={"human_nominal": 0.5, "human_corrective": 0.25, "robot_nominal_video": 0.25})
    return replace(base, **overrides)

import hashlib
from dataclasses import replace

import torch
from torch import nn

from twe.config import Stage1Config, WorldConfig
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


class StubText(nn.Module):
    def __init__(self, dim: int = 512, length: int = 64):
        super().__init__()
        torch.manual_seed(1)
        self.embed = nn.Embedding(997, dim)
        self.length = length
        self.revision = "stub-text"

    def forward(self, instructions):
        device = self.embed.weight.device
        ids = torch.zeros(len(instructions), self.length, dtype=torch.long, device=device)
        mask = torch.zeros(len(instructions), self.length, dtype=torch.bool, device=device)
        for row, text in enumerate(instructions):
            for col, word in enumerate((text or "").split()):
                ids[row, col] = int(hashlib.md5(word.encode()).hexdigest(), 16) % 997
                mask[row, col] = True
        null = torch.tensor([not t for t in instructions], device=device)
        return self.embed(ids) * mask[..., None], mask, null


def tiny_module(cfg: WorldConfig = TINY_WORLD) -> WorldModule:
    return WorldModule(cfg, StubVisual(), StubText())


def tiny_stage1(**overrides) -> Stage1Config:
    base = Stage1Config(world=TINY_WORLD, micro_batch_size=8, effective_batch_size=16, optimizer_updates=60,
                        learning_rate=1e-3, warmup_fraction=0.05, num_workers=0, log_every=10, eval_every=30,
                        eval_batches=2, checkpoint_every=30,
                        mixture={"human_nominal": 0.5, "human_corrective": 0.25, "robot_nominal_video": 0.25})
    return replace(base, **overrides)

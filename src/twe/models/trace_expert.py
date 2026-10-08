from dataclasses import dataclass

import torch
from torch import Tensor, nn

from twe.config import WorldConfig
from twe.models.blocks import Head, TraceBlock
from twe.models.embeddings import TimeEmbedding, fourier_uv, sincos_2d


@dataclass
class ExpertInputs:
    visual: Tensor
    visual_mask: Tensor
    text: Tensor
    text_mask: Tensor
    text_null: Tensor
    anchor_features: Tensor
    anchor_uv: Tensor
    anchor_mask: Tensor
    history: Tensor
    history_valid: Tensor
    grounded: Tensor
    grounded_mask: Tensor
    grounded_local: Tensor
    alignment: Tensor
    fused_local: Tensor | None = None


@dataclass
class ExpertOutputs:
    hidden: Tensor
    velocity: Tensor
    validity_logits: Tensor
    motion_logits: Tensor


class TraceExpert(nn.Module):
    def __init__(self, cfg: WorldConfig):
        super().__init__()
        width = cfg.width
        self.cfg = cfg
        self.visual_proj = nn.Linear(cfg.text_dim if cfg.fusion_layers else cfg.visual_dim, width)
        self.fused_local_proj = nn.Linear(cfg.text_dim, width) if cfg.fusion_layers else None
        self.text_proj = nn.Linear(cfg.text_dim, width)
        self.grounded_proj = nn.Linear(cfg.text_dim, width)
        self.grounded_type = nn.Parameter(torch.zeros(width))
        self.local_grounding_proj = nn.Linear(cfg.text_dim + 1, width)
        self.register_buffer("visual_pos", sincos_2d(cfg.pooled_grid, width), persistent=False)
        self.visual_type = nn.Parameter(torch.zeros(width))
        self.text_type = nn.Parameter(torch.zeros(width))
        self.null_text = nn.Parameter(torch.zeros(width))
        self.control_proj = nn.Linear(cfg.free_control_points * 3, width)
        self.feature_proj = nn.Linear(cfg.visual_dim, width)
        self.uv_proj = nn.Linear(2 + 4 * cfg.uv_frequencies, width)
        self.history_proj = nn.Sequential(nn.Linear(cfg.history_steps * 4, width), nn.GELU(), nn.Linear(width, width))
        self.no_history = nn.Parameter(torch.zeros(width))
        self.time_embed = TimeEmbedding(cfg.time_embedding_dim, width)
        self.blocks = nn.ModuleList(TraceBlock(width, cfg.heads, cfg.ffn_width, cfg.dropout) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(width)
        self.velocity_head = Head(width, cfg.free_control_points * 3)
        self.validity_head = Head(width, cfg.future_steps)
        self.motion_head = Head(width, 1)
        for p in (self.visual_type, self.text_type, self.null_text, self.no_history, self.grounded_type):
            nn.init.normal_(p, std=0.02)

    def context(self, inputs: ExpertInputs) -> tuple[Tensor, Tensor]:
        visual = self.visual_proj(inputs.visual) + self.visual_pos + self.visual_type
        text = self.text_proj(inputs.text) + self.text_type
        null = inputs.text_null[:, None]
        first = torch.where(null, self.null_text.to(text.dtype).expand_as(text[:, 0]), text[:, 0])
        text = torch.cat([first[:, None], text[:, 1:]], dim=1)
        text_mask = inputs.text_mask.clone()
        text_mask[:, 0] |= inputs.text_null
        grounded = self.grounded_proj(inputs.grounded) + self.grounded_type
        return (torch.cat([visual, grounded, text], dim=1),
                torch.cat([inputs.visual_mask, inputs.grounded_mask, text_mask], dim=1))

    def history_embedding(self, inputs: ExpertInputs) -> Tensor:
        valid = inputs.history_valid.to(inputs.history.dtype)
        features = torch.cat([inputs.history * valid[..., None], valid[..., None]], -1).flatten(2)
        seen = inputs.history_valid.any(-1, keepdim=True)
        return torch.where(seen, self.history_proj(features), self.no_history.to(features.dtype))

    def point_features(self, inputs: ExpertInputs) -> Tensor:
        x = (self.feature_proj(inputs.anchor_features)
             + self.uv_proj(fourier_uv(inputs.anchor_uv, self.cfg.uv_frequencies))
             + self.history_embedding(inputs)
             + self.local_grounding_proj(torch.cat([inputs.grounded_local, inputs.alignment[..., None]], -1)))
        if self.fused_local_proj is not None:
            x = x + self.fused_local_proj(inputs.fused_local)
        return x

    def forward(self, inputs: ExpertInputs, noisy_controls: Tensor, s: Tensor) -> ExpertOutputs:
        batch, points = noisy_controls.shape[:2]
        context, context_mask = self.context(inputs)
        x = (
            self.control_proj(noisy_controls.flatten(2))
            + self.point_features(inputs)
            + self.time_embed(s)[:, None]
        ).float()
        for block in self.blocks:
            x = block(x, inputs.anchor_mask, context, context_mask)
        hidden = self.norm(x)
        velocity = self.velocity_head(hidden).view(batch, points, self.cfg.free_control_points, 3)
        return ExpertOutputs(hidden, velocity, self.validity_head(hidden), self.motion_head(hidden)[..., 0])

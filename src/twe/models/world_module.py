import torch
from torch import Tensor, nn

from twe.config import WorldConfig
from twe.contracts import NOISE_PROTOCOL, PREPROCESSING_REVISION, WorldContext, WorldFeatures
from twe.models.grounding_encoder import alignment_scores, map_validity
from twe.models.trace_expert import ExpertInputs, ExpertOutputs, TraceExpert
from twe.preprocess.current_anchors import patch_validity, pool_visual, sample_anchor_features, select_anchors


class WorldModule(nn.Module):
    def __init__(self, cfg: WorldConfig, visual_encoder: nn.Module, grounding_encoder: nn.Module,
                 fusion: nn.Module | None = None):
        super().__init__()
        if bool(cfg.fusion_layers) != (fusion is not None):
            raise ValueError("fusion module must be given exactly when fusion_layers > 0")
        self.cfg = cfg
        self.visual_encoder = visual_encoder.requires_grad_(False)
        self.grounding_encoder = grounding_encoder.requires_grad_(False)
        self.fusion = fusion
        self.expert = TraceExpert(cfg)
        self.model_revision = "untrained"

    def trainable(self) -> dict[str, nn.Module]:
        parts = {"expert": self.expert}
        if self.fusion is not None:
            parts["fusion"] = self.fusion
        return parts

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for part in self.trainable().values() for p in part.parameters() if p.requires_grad]

    def trainable_state(self) -> dict[str, dict]:
        return {name: part.state_dict() for name, part in self.trainable().items()}

    def load_trainable_state(self, state: dict[str, dict]) -> None:
        if set(state) != set(self.trainable()):
            raise ValueError(f"trainable parts {sorted(state)} != {sorted(self.trainable())}")
        for name, part in self.trainable().items():
            part.load_state_dict(state[name])

    def encode(self, context: WorldContext) -> ExpertInputs:
        with torch.no_grad():
            patches = self.visual_encoder(context.rgb).float()
            grounding = self.grounding_encoder(context.rgb, context.image_valid, context.instructions)
        weight = patch_validity(context.image_valid, self.cfg.patch_grid)
        text, text_mask, fused_local = grounding.text, grounding.text_mask, None
        if self.fusion is None:
            visual, visual_mask = pool_visual(patches, weight, self.cfg.pooled_grid)
        else:
            fused = self.fusion(patches, weight, grounding.prompt)
            visual, visual_mask = pool_visual(fused.visual, weight, self.cfg.pooled_grid)
            text, text_mask = fused.text, grounding.prompt.mask
            fused_local = sample_anchor_features(fused.visual, context.anchor_uv)
        anchors = sample_anchor_features(patches, context.anchor_uv)
        fine = grounding.maps[0].permute(0, 2, 3, 1)
        grounded_local = sample_anchor_features(fine, context.anchor_uv)
        alignment = alignment_scores(grounded_local, grounding.text, grounding.text_mask, grounding.null)
        coarse = grounding.maps[self.cfg.grounding_context_level]
        grounded = coarse.flatten(2).transpose(1, 2)
        grounded_mask = map_validity(context.image_valid, tuple(coarse.shape[-2:])).flatten(1)
        history, history_valid = self.history_inputs(context)
        return ExpertInputs(visual, visual_mask, text, text_mask, grounding.null,
                            anchors, context.anchor_uv, context.anchor_mask, history, history_valid,
                            grounded, grounded_mask, grounded_local, alignment, fused_local)

    def history_inputs(self, context: WorldContext) -> tuple[Tensor, Tensor]:
        if context.history is not None:
            return context.history.float(), context.history_valid
        batch, points = context.anchor_uv.shape[:2]
        steps = self.cfg.history_steps
        device = context.anchor_uv.device
        return (torch.zeros(batch, points, steps, 3, device=device),
                torch.zeros(batch, points, steps, dtype=torch.bool, device=device))

    def forward(self, inputs: ExpertInputs, noisy_controls: Tensor, s: Tensor) -> ExpertOutputs:
        return self.expert(inputs, noisy_controls, s)

    def guidance_noise(self, batch: int, device: torch.device) -> Tensor:
        generator = torch.Generator().manual_seed(self.cfg.guidance_noise_seed)
        shape = (self.cfg.num_anchors, self.cfg.free_control_points, 3)
        return torch.randn(shape, generator=generator).to(device).expand(batch, *shape)

    def select_anchors(self, rgb: Tensor, image_valid: Tensor) -> tuple[Tensor, Tensor]:
        with torch.no_grad():
            patches = self.visual_encoder(rgb).float()
        return select_anchors(patches, image_valid, self.cfg.num_anchors, self.cfg.anchor_entities,
                              self.cfg.anchor_min_per_entity, self.cfg.anchor_spatial_weight,
                              self.cfg.anchor_area_power, self.cfg.anchor_foreground_fraction)

    @torch.no_grad()
    def extract_features(self, rgb: Tensor, image_valid: Tensor, instructions: list[str | None]) -> WorldFeatures:
        uv, mask = self.select_anchors(rgb, image_valid)
        inputs = self.encode(WorldContext(rgb, image_valid, uv, mask, instructions))
        noise = self.guidance_noise(rgb.shape[0], rgb.device)
        s = torch.full((rgb.shape[0],), self.cfg.guidance_time, device=rgb.device)
        hidden = self.expert(inputs, noise, s).hidden
        return WorldFeatures(hidden, uv, mask, self.model_revision, PREPROCESSING_REVISION, NOISE_PROTOCOL)

    def guided_outputs(self, inputs: ExpertInputs, controls: Tensor, s: Tensor, guidance: float,
                       null_inputs: ExpertInputs | None) -> tuple[Tensor, Tensor]:
        out = self.expert(inputs, controls, s)
        velocity, logits = out.velocity.float(), out.motion_logits.float()
        if null_inputs is None or guidance == 1.0:
            return velocity, logits
        base = self.expert(null_inputs, controls, s)
        return (base.velocity.float() + guidance * (velocity - base.velocity.float()),
                base.motion_logits.float() + guidance * (logits - base.motion_logits.float()))

    @torch.no_grad()
    def sample_controls(self, inputs: ExpertInputs, steps: int, generator: torch.Generator | None = None,
                        guidance: float = 1.0, null_inputs: ExpertInputs | None = None) -> Tensor:
        batch, points = inputs.anchor_uv.shape[:2]
        shape = (batch, points, self.cfg.free_control_points, 3)
        controls = torch.randn(shape, generator=generator).to(inputs.anchor_uv.device)
        times = torch.linspace(1.0, 0.0, steps + 1, device=controls.device)
        for start, end in zip(times[:-1], times[1:]):
            s = torch.full((batch,), float(start), device=controls.device)
            velocity, _ = self.guided_outputs(inputs, controls, s, guidance, null_inputs)
            controls = controls - (start - end) * velocity
        return controls

    @torch.no_grad()
    def motion_probability(self, inputs: ExpertInputs, guidance: float = 1.0,
                           null_inputs: ExpertInputs | None = None) -> Tensor:
        batch, points = inputs.anchor_uv.shape[:2]
        noise = self.guidance_noise(batch, inputs.anchor_uv.device)[:, :points]
        s = torch.ones(batch, device=inputs.anchor_uv.device)
        return torch.sigmoid(self.guided_outputs(inputs, noise, s, guidance, null_inputs)[1])


def build_world_module(cfg: WorldConfig) -> WorldModule:
    from twe.models.fusion import VisionLanguageFusion
    from twe.models.grounding_encoder import GroundingDinoEncoder
    from twe.models.visual_encoder import DinoVisualEncoder

    visual = DinoVisualEncoder(cfg.visual_encoder, cfg.visual_encoder_revision, cfg.patch_grid)
    grounding = GroundingDinoEncoder(cfg.text_encoder, cfg.text_encoder_revision, cfg.text_max_length)
    fusion = None
    if cfg.fusion_layers:
        fusion = VisionLanguageFusion(grounding.fusion_layers(cfg.fusion_layers), cfg.visual_dim, cfg.text_dim,
                                      cfg.patch_grid, cfg.dropout)
    return WorldModule(cfg, visual, grounding, fusion)

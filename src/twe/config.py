from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class WorldConfig:
    visual_encoder: str = "facebook/dinov2-small"
    visual_encoder_revision: str = "main"
    text_encoder: str = "google-t5/t5-small"
    text_encoder_revision: str = "main"
    image_size: int = 224
    patch_grid: int = 16
    visual_dim: int = 384
    pooled_grid: int = 8
    text_dim: int = 512
    text_max_length: int = 64
    num_anchors: int = 64
    query_pool: int = 128
    anchor_entities: int = 12
    anchor_min_per_entity: int = 3
    anchor_spatial_weight: float = 0.5
    anchor_area_power: float = 0.35
    anchor_foreground_fraction: float = 0.75
    width: int = 512
    layers: int = 8
    heads: int = 8
    ffn_width: int = 2048
    dropout: float = 0.0
    horizon_seconds: float = 2.0
    future_steps: int = 32
    bspline_degree: int = 3
    free_control_points: int = 10
    uv_frequencies: int = 8
    time_embedding_dim: int = 256
    guidance_time: float = 1.0
    guidance_noise_seed: int = 0
    trace_eval_solver_steps: int = 4

    @property
    def future_offsets(self) -> list[float]:
        step = self.horizon_seconds / self.future_steps
        return [step * k for k in range(1, self.future_steps + 1)]


@dataclass(frozen=True)
class Stage1Config:
    world: WorldConfig
    optimizer: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    gradient_clip_norm: float = 1.0
    precision: str = "bfloat16"
    effective_batch_size: int = 128
    micro_batch_size: int = 64
    optimizer_updates: int = 20000
    warmup_fraction: float = 0.05
    endpoint_time_probability: float = 0.2
    max_null_text_fraction: float = 0.2
    validity_loss_weight: float = 0.1
    fit_regularization: float = 1e-3
    fit_min_valid_steps: int = 10
    fit_max_condition: float = 1e5
    reliability_threshold: float = 0.5
    min_moving_points: int = 4
    mixture: dict[str, float] = field(default_factory=dict)
    num_workers: int = 8
    log_every: int = 50
    eval_every: int = 1000
    eval_batches: int = 20
    checkpoint_every: int = 1000
    seed: int = 0

    @property
    def accumulation_steps(self) -> int:
        if self.effective_batch_size % self.micro_batch_size:
            raise ValueError("effective_batch_size must be divisible by micro_batch_size")
        return self.effective_batch_size // self.micro_batch_size


def _build(cls, values: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**values)


def load_world_config(path: str | Path) -> WorldConfig:
    return _build(WorldConfig, yaml.safe_load(Path(path).read_text()))


def load_stage1_config(path: str | Path, overrides: dict | None = None) -> Stage1Config:
    path = Path(path)
    values = yaml.safe_load(path.read_text())
    overrides = dict(overrides or {})
    world_overrides = {k.split(".", 1)[1]: overrides.pop(k) for k in list(overrides) if k.startswith("world.")}
    values.update(overrides)
    world_ref = values.pop("world")
    world_path = Path(world_ref)
    if not world_path.is_absolute() and not world_path.exists():
        world_path = path.parent.parent / world_ref
    world = yaml.safe_load(world_path.read_text())
    world.update(world_overrides)
    values["world"] = _build(WorldConfig, world)
    values["betas"] = tuple(values.get("betas", (0.9, 0.95)))
    return _build(Stage1Config, values)

from dataclasses import dataclass

import torch
from torch import Tensor

COORDINATE_CONTRACT = "camera_t_relative_xyz_v1"
PREPROCESSING_REVISION = "letterbox224_dinogrid16_entitypool128_moving_v3"
NOISE_PROTOCOL = "fixed_seed_gaussian_s1_v1"
POOLS = ("human_nominal", "human_corrective", "robot_nominal_video")


@dataclass
class LetterboxTransform:
    scale_x: float
    scale_y: float
    offset_x: float
    offset_y: float
    source_width: int
    source_height: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class WorldContext:
    rgb: Tensor
    image_valid: Tensor
    anchor_uv: Tensor
    anchor_mask: Tensor
    instructions: list[str | None]
    history: Tensor | None = None
    history_valid: Tensor | None = None

    def without_history(self) -> "WorldContext":
        return WorldContext(self.rgb, self.image_valid, self.anchor_uv, self.anchor_mask, self.instructions)

    def to(self, device: torch.device) -> "WorldContext":
        move = lambda t: None if t is None else t.to(device, non_blocking=True)
        return WorldContext(
            self.rgb.to(device, non_blocking=True),
            self.image_valid.to(device, non_blocking=True),
            self.anchor_uv.to(device, non_blocking=True),
            self.anchor_mask.to(device, non_blocking=True),
            self.instructions,
            move(self.history),
            move(self.history_valid),
        )


@dataclass
class WorldTarget:
    controls: Tensor
    fit_valid: Tensor
    trace: Tensor
    trace_valid: Tensor
    moving: Tensor

    def to(self, device: torch.device) -> "WorldTarget":
        return WorldTarget(
            self.controls.to(device, non_blocking=True),
            self.fit_valid.to(device, non_blocking=True),
            self.trace.to(device, non_blocking=True),
            self.trace_valid.to(device, non_blocking=True),
            self.moving.to(device, non_blocking=True),
        )


@dataclass
class WorldFeatures:
    hidden: Tensor
    anchor_uv: Tensor
    anchor_mask: Tensor
    model_revision: str
    preprocessing_revision: str
    noise_protocol: str

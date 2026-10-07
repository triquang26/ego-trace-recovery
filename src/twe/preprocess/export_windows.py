from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from twe.config import WorldConfig
from twe.contracts import COORDINATE_CONTRACT
from twe.data.shards import ShardWriter
from twe.preprocess.camera_reference import (relative_displacements, robust_scene_scale, to_opencv_camera,
                                             world_to_reference_camera)
from twe.preprocess.letterbox import letterbox, uv_to_source_xy
from twe.preprocess.teacher import TrackTeacher
from twe.preprocess.temporal_sampling import gather_tracks, plan_future_samples

AnchorSelector = Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class Recording:
    recording_id: str
    source: str
    split_group: str
    instruction: str | None
    timestamps: np.ndarray
    read_frames: Callable[[np.ndarray], np.ndarray]
    metadata: dict = field(default_factory=dict)


@dataclass
class ExportSettings:
    max_gap: float = 0.1
    tolerance: float = 0.004
    reliability_threshold: float = 0.5
    geometry_provenance: str = "estimated"


def export_window(recording: Recording, current_index: int, teacher: TrackTeacher, select: AnchorSelector,
                  cfg: WorldConfig, settings: ExportSettings) -> tuple[dict, dict] | None:
    times = recording.timestamps
    current_time = float(times[current_index])
    horizon_end = current_time + cfg.horizon_seconds + settings.max_gap
    indices = np.arange(current_index, int(np.searchsorted(times, horizon_end, side="right")))
    frames = recording.read_frames(indices)
    rgb, valid, transform = letterbox(frames[0], cfg.image_size)
    rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255.0
    uv, mask = select(rgb_tensor, torch.from_numpy(valid)[None])
    uv, mask = uv[0].cpu().numpy(), mask[0].cpu().numpy()
    query_xy = uv_to_source_xy(uv, transform, cfg.image_size)
    tracks = teacher.track(frames, query_xy)
    scale = robust_scene_scale(tracks.current_depth)
    if scale is None:
        return None
    window_times = times[indices] - current_time
    plan = plan_future_samples(window_times, 0.0, [0.0] + cfg.future_offsets, settings.max_gap, settings.tolerance)
    points, reliability = gather_tracks(tracks.points_world, tracks.reliability, plan)
    camera = world_to_reference_camera(points, tracks.world_to_camera[0])
    camera = to_opencv_camera(camera, tracks.convention)
    trace, trace_valid, trace_reliability = relative_displacements(camera, reliability, scale,
                                                                   settings.reliability_threshold)
    trace_valid &= mask[:, None]
    arrays = {"rgb": rgb, "image_valid": valid, "anchor_uv": uv, "anchor_mask": mask, "trace": trace,
              "trace_valid": trace_valid, "trace_reliability": trace_reliability * trace_valid}
    meta = {
        "sample_id": f"{recording.source}/{recording.recording_id}/{current_time:.3f}",
        "source": recording.source,
        "recording_id": recording.recording_id,
        "split_group": recording.split_group,
        "original_instruction": recording.instruction,
        "instruction_available_at_t": recording.instruction is not None,
        "current_timestamp_seconds": current_time,
        "future_offsets_seconds": cfg.future_offsets,
        "letterbox_transform": transform.to_dict(),
        "scene_scale": scale,
        "geometry_provenance": settings.geometry_provenance,
        "teacher_revision": tracks.revision,
        "coordinate_contract": COORDINATE_CONTRACT,
        **recording.metadata,
    }
    return arrays, meta


def export_recording(recording: Recording, current_indices: list[int], teacher: TrackTeacher,
                     select: AnchorSelector, cfg: WorldConfig, settings: ExportSettings, writer: ShardWriter) -> int:
    written = 0
    for index in current_indices:
        result = export_window(recording, index, teacher, select, cfg, settings)
        if result is not None:
            writer.add(*result)
            written += 1
    return written


def window_starts(timestamps: np.ndarray, horizon: float, stride: float) -> list[int]:
    starts, next_time = [], float(timestamps[0])
    last_start = float(timestamps[-1]) - horizon
    for index, time in enumerate(timestamps):
        if time > last_start:
            break
        if time >= next_time:
            starts.append(index)
            next_time = time + stride
    return starts

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from twe.config import WorldConfig
from twe.contracts import COORDINATE_CONTRACT
from twe.data.shards import ShardWriter
from twe.preprocess.camera_reference import (moving_points, relative_displacements, robust_scene_scale,
                                             to_opencv_camera, world_to_reference_camera)
from twe.preprocess.letterbox import letterbox, uv_to_source_xy
from twe.preprocess.chunking import Chunk, plan_chunks, window_starts
from twe.preprocess.teacher import TeacherTracks, TrackTeacher
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
    chunk_seconds: float = 10.0
    moving_threshold_px: float = 35.0


def letterbox_intrinsics(intrinsics: np.ndarray | None, transform) -> np.ndarray:
    if intrinsics is None:
        return np.full((3, 3), np.nan, dtype=np.float32)
    lift = np.array([[transform.scale_x, 0, transform.offset_x], [0, transform.scale_y, transform.offset_y], [0, 0, 1]])
    return (lift @ intrinsics).astype(np.float32)


def anchor_queries(frames: np.ndarray, starts: list[int], select: AnchorSelector, size: int) -> list[dict]:
    views = [letterbox(frames[s], size) for s in starts]
    rgb = torch.stack([torch.from_numpy(v[0]).permute(2, 0, 1) for v in views]).float() / 255.0
    valid = torch.stack([torch.from_numpy(v[1]) for v in views])
    uv, mask = select(rgb, valid)
    uv, mask = uv.cpu().numpy(), mask.cpu().numpy()
    return [{"start": s, "rgb": v[0], "valid": v[1], "transform": v[2], "uv": uv[i], "mask": mask[i],
             "xy": uv_to_source_xy(uv[i], v[2], size)} for i, (s, v) in enumerate(zip(starts, views))]


def window_record(recording: Recording, chunk: Chunk, query: dict, rows: slice, tracks: TeacherTracks,
                  cfg: WorldConfig, settings: ExportSettings) -> tuple[dict, dict] | None:
    s = query["start"]
    scale = robust_scene_scale(tracks.depth[s])
    if scale is None:
        return None
    times = recording.timestamps[chunk.frames]
    current_time = float(times[s])
    plan = plan_future_samples(times[s:] - current_time, 0.0, [0.0] + cfg.future_offsets, settings.max_gap,
                               settings.tolerance)
    points, reliability = gather_tracks(tracks.points_world[rows, s:], tracks.reliability[rows, s:], plan)
    camera = to_opencv_camera(world_to_reference_camera(points, tracks.world_to_camera[s]), tracks.convention)
    trace, trace_valid, trace_reliability = relative_displacements(camera, reliability, scale,
                                                                   settings.reliability_threshold)
    mask = query["mask"]
    trace_valid &= mask[:, None]
    anchor_xyz = np.where(np.isfinite(camera[:, 0]), camera[:, 0] / scale, 0.0).astype(np.float32)
    k = None if tracks.intrinsics is None else tracks.intrinsics[s]
    intrinsics = letterbox_intrinsics(k, query["transform"])
    moving = moving_points(camera, trace_valid, intrinsics, settings.moving_threshold_px) & mask
    arrays = {"anchor_xyz": anchor_xyz, "intrinsics": intrinsics, "trace_moving": moving,
              "rgb": query["rgb"], "image_valid": query["valid"], "anchor_uv": query["uv"], "anchor_mask": mask,
              "trace": trace, "trace_valid": trace_valid, "trace_reliability": trace_reliability * trace_valid}
    meta = {
        "sample_id": f"{recording.source}/{recording.recording_id}/{current_time:.3f}",
        "source": recording.source,
        "recording_id": recording.recording_id,
        "split_group": recording.split_group,
        "original_instruction": recording.instruction,
        "instruction_available_at_t": recording.instruction is not None,
        "current_timestamp_seconds": current_time,
        "future_offsets_seconds": cfg.future_offsets,
        "letterbox_transform": query["transform"].to_dict(),
        "scene_scale": scale,
        "geometry_provenance": settings.geometry_provenance,
        "teacher_revision": tracks.revision,
        "coordinate_contract": COORDINATE_CONTRACT,
        **recording.metadata,
    }
    return arrays, meta


def export_chunk(recording: Recording, chunk: Chunk, teacher: TrackTeacher, select: AnchorSelector,
                 cfg: WorldConfig, settings: ExportSettings) -> list[tuple[dict, dict]]:
    frames = recording.read_frames(chunk.frames)
    queries = anchor_queries(frames, chunk.starts, select, cfg.image_size)
    xy = np.concatenate([q["xy"] for q in queries])
    when = np.concatenate([np.full(len(q["xy"]), q["start"]) for q in queries])
    tracks = teacher.track(frames, xy, when)
    count = cfg.num_anchors
    records = [window_record(recording, chunk, q, slice(i * count, (i + 1) * count), tracks, cfg, settings)
               for i, q in enumerate(queries)]
    return [r for r in records if r is not None]


def export_recording(recording: Recording, teacher: TrackTeacher, select: AnchorSelector, cfg: WorldConfig,
                     settings: ExportSettings, writer: ShardWriter, stride: float) -> int:
    span = cfg.horizon_seconds + settings.max_gap
    starts = window_starts(recording.timestamps, span, stride)
    written = 0
    for chunk in plan_chunks(recording.timestamps, starts, span, settings.chunk_seconds):
        for arrays, meta in export_chunk(recording, chunk, teacher, select, cfg, settings):
            writer.add(arrays, meta)
            written += 1
    return written

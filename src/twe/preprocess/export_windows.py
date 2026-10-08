from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from twe.config import WorldConfig
from twe.contracts import COORDINATE_CONTRACT
from twe.data.shards import ShardWriter
from twe.preprocess.camera_reference import (moving_points, relative_displacements, robust_scene_scale,
                                             to_opencv_camera, world_to_reference_camera)
from twe.preprocess.chunking import Chunk, plan_chunks, window_starts
from twe.preprocess.letterbox import letterbox, uv_to_source_xy
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
    moving_threshold_px: float = 10.0


def letterbox_intrinsics(intrinsics: np.ndarray | None, transform) -> np.ndarray:
    if intrinsics is None:
        return np.full((3, 3), np.nan, dtype=np.float32)
    lift = np.array([[transform.scale_x, 0, transform.offset_x], [0, transform.scale_y, transform.offset_y], [0, 0, 1]])
    return (lift @ intrinsics).astype(np.float32)


def window_span(cfg: WorldConfig, settings: ExportSettings) -> float:
    return -cfg.history_offsets[0] + cfg.horizon_seconds + settings.max_gap


def anchor_queries(frames: np.ndarray, starts: list[int], select: AnchorSelector, size: int) -> list[dict]:
    views = [letterbox(frames[s], size) for s in starts]
    rgb = torch.stack([torch.from_numpy(v[0]).permute(2, 0, 1) for v in views]).float() / 255.0
    valid = torch.stack([torch.from_numpy(v[1]) for v in views])
    uv, mask = select(rgb, valid)
    uv, mask = uv.cpu().numpy(), mask.cpu().numpy()
    return [{"start": s, "mask": mask[i], "xy": uv_to_source_xy(uv[i], v[2], size)}
            for i, (s, v) in enumerate(zip(starts, views))]


def project_uv(points: np.ndarray, intrinsics: np.ndarray, size: int) -> np.ndarray:
    z = np.where(points[:, 2] > 0, points[:, 2], np.nan)
    u = (intrinsics[0, 0] * points[:, 0] / z + intrinsics[0, 2]) / size
    v = (intrinsics[1, 1] * points[:, 1] / z + intrinsics[1, 2]) / size
    return np.stack([u, v], -1)


def window_record(recording: Recording, chunk: Chunk, frames: np.ndarray, query: dict, rows: slice,
                  tracks: TeacherTracks, cfg: WorldConfig, settings: ExportSettings) -> tuple[dict, dict] | None:
    times = recording.timestamps[chunk.frames]
    q = query["start"]
    current = int(np.searchsorted(times, times[q] - cfg.history_offsets[0] - settings.tolerance))
    scale = robust_scene_scale(tracks.depth[current])
    if scale is None:
        return None
    current_time = float(times[current])
    history = len(cfg.history_offsets)
    offsets = cfg.history_offsets + [0.0] + cfg.future_offsets
    plan = plan_future_samples(times[q:] - current_time, 0.0, offsets, settings.max_gap, settings.tolerance)
    points, reliability = gather_tracks(tracks.points_world[rows, q:], tracks.reliability[rows, q:], plan)
    camera = to_opencv_camera(world_to_reference_camera(points, tracks.world_to_camera[current]), tracks.convention)
    trace, valid, trust = relative_displacements(camera, reliability, scale, settings.reliability_threshold, history)
    rgb, image_valid, transform = letterbox(frames[current], cfg.image_size)
    k = None if tracks.intrinsics is None else tracks.intrinsics[current]
    intrinsics = letterbox_intrinsics(k, transform)
    uv = project_uv(camera[:, history], intrinsics, cfg.image_size)
    pixel = np.clip(np.nan_to_num(uv * cfg.image_size, nan=-1).astype(int), -1, cfg.image_size)
    inside = (pixel >= 0).all(-1) & (pixel < cfg.image_size).all(-1)
    inside &= image_valid[np.clip(pixel[:, 1], 0, cfg.image_size - 1), np.clip(pixel[:, 0], 0, cfg.image_size - 1)]
    mask = query["mask"] & valid[:, history] & inside
    future, future_valid = trace[:, history + 1 :], valid[:, history + 1 :] & mask[:, None]
    moving = moving_points(camera[:, history:], future_valid, intrinsics, settings.moving_threshold_px) & mask
    arrays = {"rgb": rgb, "image_valid": image_valid, "anchor_uv": np.nan_to_num(uv).astype(np.float32),
              "anchor_mask": mask, "anchor_xyz": np.nan_to_num(camera[:, history] / scale).astype(np.float32),
              "intrinsics": intrinsics, "trace": future, "trace_valid": future_valid,
              "trace_reliability": trust[:, history + 1 :] * future_valid, "trace_moving": moving,
              "history": trace[:, :history], "history_valid": valid[:, :history] & mask[:, None]}
    meta = {
        "sample_id": f"{recording.source}/{recording.recording_id}/{current_time:.3f}",
        "source": recording.source,
        "recording_id": recording.recording_id,
        "split_group": recording.split_group,
        "original_instruction": recording.instruction,
        "instruction_available_at_t": recording.instruction is not None,
        "current_timestamp_seconds": current_time,
        "future_offsets_seconds": cfg.future_offsets,
        "history_offsets_seconds": cfg.history_offsets,
        "letterbox_transform": transform.to_dict(),
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
    count = len(queries[0]["xy"])
    records = [window_record(recording, chunk, frames, q, slice(i * count, (i + 1) * count), tracks, cfg, settings)
               for i, q in enumerate(queries)]
    return [r for r in records if r is not None]


def export_chunk_safely(recording: Recording, chunk: Chunk, teacher: TrackTeacher, select: AnchorSelector,
                        cfg: WorldConfig, settings: ExportSettings) -> list[tuple[dict, dict]]:
    try:
        return export_chunk(recording, chunk, teacher, select, cfg, settings)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(chunk.starts) < 2:
            raise
    starts = [int(chunk.frames[s]) for s in chunk.starts]
    middle = len(starts) // 2
    records = []
    for part in (starts[:middle], starts[middle:]):
        (piece,) = plan_chunks(recording.timestamps, part, window_span(cfg, settings), float("inf"))
        records += export_chunk_safely(recording, piece, teacher, select, cfg, settings)
    return records


def export_recording(recording: Recording, teacher: TrackTeacher, select: AnchorSelector, cfg: WorldConfig,
                     settings: ExportSettings, writer: ShardWriter, stride: float) -> int:
    span = window_span(cfg, settings)
    starts = window_starts(recording.timestamps, span, stride)
    written = 0
    for chunk in plan_chunks(recording.timestamps, starts, span, settings.chunk_seconds):
        for arrays, meta in export_chunk_safely(recording, chunk, teacher, select, cfg, settings):
            writer.add(arrays, meta)
            written += 1
    return written

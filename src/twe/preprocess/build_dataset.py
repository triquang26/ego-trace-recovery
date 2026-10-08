import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from twe.config import WorldConfig
from twe.data.manifest import Manifest
from twe.data.shards import ShardReader, ShardWriter
from twe.preprocess.camera_reference import moving_points
from twe.preprocess.current_anchors import select_anchors
from twe.preprocess.export_windows import ExportSettings, Recording, export_recording
from twe.preprocess.teacher import TrackTeacher


def dino_selector(visual_encoder: torch.nn.Module, cfg: WorldConfig, device: str):
    def select(rgb: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            patches = visual_encoder(rgb.to(device)).float()
        return select_anchors(patches, valid.to(device), cfg.query_pool, cfg.anchor_entities,
                              cfg.anchor_min_per_entity, cfg.anchor_spatial_weight, cfg.anchor_area_power,
                              cfg.anchor_foreground_fraction)

    return select


def export_shard(recordings: Iterable[Recording], teacher: TrackTeacher, select, cfg: WorldConfig,
                 settings: ExportSettings, root: Path, name: str, pool: str, split: str, stride: float) -> dict:
    root = Path(root)
    if (root / name / "entry.json").exists():
        return json.loads((root / name / "entry.json").read_text())
    writer = ShardWriter(root, name, pool, split)
    report = {"recordings": 0, "skipped": [], "windows": 0}
    for recording in recordings:
        try:
            report["windows"] += export_recording(recording, teacher, select, cfg, settings, writer, stride)
            report["recordings"] += 1
        except (ValueError, torch.OutOfMemoryError) as error:
            torch.cuda.empty_cache()
            report["skipped"].append({"recording": recording.recording_id, "error": str(error)[:300]})
        print(json.dumps({"shard": name, **{k: v for k, v in report.items() if k != "skipped"},
                          "skipped": len(report["skipped"])}), flush=True)
    if len(writer):
        writer.close()
    (root / f"{name}.report.json").write_text(json.dumps(report, indent=2))
    return report


def relabel_moving(root: Path, threshold_px: float) -> int:
    moving_total = 0
    for entry in Manifest.read(root).shards:
        reader = ShardReader(root, entry)
        arrays = {key: np.asarray(reader.arrays[key]) for key in ("anchor_xyz", "trace", "trace_valid", "intrinsics",
                                                                  "anchor_mask")}
        labels = np.stack([
            moving_points(np.concatenate([arrays["anchor_xyz"][w][:, None],
                                          arrays["anchor_xyz"][w][:, None] + arrays["trace"][w]], 1),
                          arrays["trace_valid"][w], arrays["intrinsics"][w], threshold_px) & arrays["anchor_mask"][w]
            for w in range(len(reader))])
        del reader
        np.save(Path(root) / entry.path / "trace_moving.npy", labels)
        moving_total += int(labels.sum())
    return moving_total


def export_egodex_shard(raw_root: Path, data_root: Path, captions_path: Path | None, part: str, split: str,
                        start: int, count: int, stride: float, every: int, frame_step: int, chunk_seconds: float,
                        device: str = "cuda") -> dict:
    from twe.data.egodex import egodex_recordings, load_captions
    from twe.models.visual_encoder import DinoVisualEncoder
    from twe.preprocess.spatracker_teacher import load_spatracker

    name = f"egodex-{part}-e{every}-{start:06d}"
    if (Path(data_root) / name / "entry.json").exists():
        return {"shard": name, "cached": True}
    cfg = WorldConfig()
    visual = DinoVisualEncoder(cfg.visual_encoder, cfg.visual_encoder_revision, cfg.patch_grid).to(device)
    captions = load_captions(captions_path) if captions_path else None
    recordings = egodex_recordings(Path(raw_root) / part, start, count, frame_step, every, captions)
    Path(data_root).mkdir(parents=True, exist_ok=True)
    report = export_shard(recordings, load_spatracker(device), dino_selector(visual, cfg, device), cfg,
                          ExportSettings(chunk_seconds=chunk_seconds), data_root, name, "human_nominal", split, stride)
    return {"shard": name, **{k: v for k, v in report.items() if k != "skipped"},
            "skipped": len(report.get("skipped", []))}

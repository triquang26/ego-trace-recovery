import json
from collections.abc import Iterable
from pathlib import Path

import torch

from twe.config import WorldConfig
from twe.data.shards import ShardWriter
from twe.preprocess.current_anchors import select_anchors
from twe.preprocess.export_windows import ExportSettings, Recording, export_recording, window_starts
from twe.preprocess.teacher import TrackTeacher


def dino_selector(visual_encoder: torch.nn.Module, cfg: WorldConfig, device: str):
    def select(rgb: torch.Tensor, valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            patches = visual_encoder(rgb.to(device)).float()
        return select_anchors(patches, valid.to(device), cfg.num_anchors, cfg.spatial_coverage_anchors,
                              cfg.anchor_min_distance)

    return select


def export_shard(recordings: Iterable[Recording], teacher: TrackTeacher, select, cfg: WorldConfig,
                 settings: ExportSettings, root: Path, name: str, pool: str, split: str, stride: float) -> dict:
    root = Path(root)
    if (root / name / "entry.json").exists():
        return json.loads((root / name / "entry.json").read_text())
    writer = ShardWriter(root, name, pool, split)
    report = {"recordings": 0, "skipped": [], "windows": 0}
    for recording in recordings:
        starts = window_starts(recording.timestamps, cfg.horizon_seconds, stride)
        try:
            report["windows"] += export_recording(recording, starts, teacher, select, cfg, settings, writer)
            report["recordings"] += 1
        except ValueError as error:
            report["skipped"].append({"recording": recording.recording_id, "error": str(error)})
        print(json.dumps({"shard": name, **{k: v for k, v in report.items() if k != "skipped"},
                          "skipped": len(report["skipped"])}), flush=True)
    if len(writer):
        writer.close()
    (root / f"{name}.report.json").write_text(json.dumps(report, indent=2))
    return report

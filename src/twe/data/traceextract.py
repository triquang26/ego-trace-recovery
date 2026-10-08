import json
from pathlib import Path

import numpy as np
import torch

from twe.config import WorldConfig
from twe.preprocess.bspline_targets import BSplineTargets
from twe.preprocess.letterbox import letterbox, source_xy_to_uv

SOURCE_FPS = {"droid": 5.0}
DEFAULT_FPS = 10.0


def traceextract_groups(root: Path) -> dict[str, list[Path]]:
    groups = {}
    for group in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        episodes = sorted(p for p in group.iterdir() if (p / "samples" / "raw_traj.npy").exists())
        if episodes:
            groups[group.name.removeprefix("test_dataset_")] = episodes
    return groups


def episode_instruction(episode: Path) -> str | None:
    texts = episode / "curated_training_texts.json"
    if texts.exists():
        chunks = json.loads(texts.read_text()).get("chunk_texts") or []
        if chunks and chunks[0].get("instruction_1"):
            return chunks[0]["instruction_1"]
    description = episode / "description.txt"
    return description.read_text().splitlines()[0] if description.exists() else None


def resample(points: np.ndarray, valid: np.ndarray, times: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    position = times * fps
    left = np.floor(position).astype(int)
    frac = (position - left)[None, :, None]
    right = left + 1
    last = points.shape[1] - 1
    out = points[:, np.minimum(left, last)] * (1 - frac) + points[:, np.minimum(right, last)] * frac
    ok = (right[None] < valid.sum(1)[:, None]) & np.isfinite(out).all(-1) & (out[..., 2] > 0)
    return out, ok


def screen_delta(points: np.ndarray, origin: np.ndarray, transform, size: int) -> np.ndarray:
    uv = source_xy_to_uv(points[..., :2], transform, size)
    uv0 = source_xy_to_uv(origin[:, None, :2], transform, size)
    depth = np.log(np.maximum(points[..., 2:3], 1e-3)) - np.log(np.maximum(origin[:, None, 2:3], 1e-3))
    return np.concatenate([uv - uv0, depth], -1)


def padded(values: np.ndarray, slots: int) -> torch.Tensor:
    out = np.zeros((slots, *values.shape[1:]), dtype=values.dtype)
    out[: len(values)] = values
    return torch.from_numpy(out)


class TraceExtractEpisode:
    def __init__(self, path: Path, fps: float):
        samples = path / "samples"
        load = lambda name: np.load(samples / f"{name}.npy", mmap_mode="r")
        self.frames = np.load(samples / "frame_indices.npy")
        self.offsets = np.load(samples / "offsets.npy")
        self.future, self.future_valid = load("raw_traj"), load("raw_valid_steps")
        self.past, self.past_valid = load("raw_traj_history"), load("raw_valid_steps_history")
        self.images = np.load(path / "images.npy", mmap_mode="r")
        self.instruction = episode_instruction(path)
        self.fps = fps

    def slots(self, minimum: int = 4) -> list[int]:
        return [i for i in range(len(self.frames)) if self.offsets[i + 1] - self.offsets[i] >= minimum]

    def item(self, slot: int, cfg: WorldConfig, fitter: BSplineTargets, sigma, rng: np.random.Generator) -> dict:
        lo, hi = int(self.offsets[slot]), int(self.offsets[slot + 1])
        rows = np.sort(rng.permutation(hi - lo)[: cfg.num_anchors]) + lo
        future = np.asarray(self.future[rows], dtype=np.float64)
        past = np.asarray(self.past[rows], dtype=np.float64)
        rgb, image_valid, transform = letterbox(np.asarray(self.images[int(self.frames[slot])]), cfg.image_size)
        step = cfg.horizon_seconds / cfg.future_steps
        origin = future[:, 0]
        ahead, ahead_ok = resample(future, np.asarray(self.future_valid[rows]),
                                   np.arange(1, cfg.future_steps + 1) * step, self.fps)
        behind, behind_ok = resample(past, np.asarray(self.past_valid[rows]),
                                     np.arange(cfg.history_steps, 0, -1) * step, self.fps)
        scale = torch.as_tensor(sigma, dtype=torch.float32)
        size, slots = cfg.image_size, cfg.num_anchors
        trace = padded(np.where(ahead_ok[..., None], screen_delta(ahead, origin, transform, size), 0.0).astype(np.float32), slots) / scale
        history = padded(np.where(behind_ok[..., None], screen_delta(behind, origin, transform, size), 0.0).astype(np.float32), slots) / scale
        valid = padded(ahead_ok, slots)
        controls, fit_valid = fitter.fit(trace, valid.float())
        mask = padded(np.ones(len(rows), dtype=bool), slots)
        return {"rgb": torch.from_numpy(rgb), "image_valid": torch.from_numpy(image_valid),
                "anchor_uv": padded(source_xy_to_uv(origin[:, :2], transform, size).astype(np.float32), slots),
                "anchor_mask": mask, "instruction": self.instruction, "trace": trace, "trace_valid": valid,
                "controls": controls.float(), "fit_valid": fit_valid & mask, "moving": mask.clone(),
                "anchor_xyz": torch.zeros(slots, 3), "history": history, "history_valid": padded(behind_ok, slots),
                "intrinsics": torch.eye(3)}


def traceextract_items(root: Path, cfg: WorldConfig, fitter: BSplineTargets, sigma, per_episode: int = 4,
                       seed: int = 0) -> dict[str, list[dict]]:
    rng = np.random.default_rng(seed)
    out = {}
    for group, episodes in traceextract_groups(root).items():
        items = []
        for path in episodes:
            episode = TraceExtractEpisode(path, SOURCE_FPS.get(group, DEFAULT_FPS))
            for slot in sorted(rng.permutation(episode.slots())[:per_episode]):
                items.append(episode.item(int(slot), cfg, fitter, sigma, rng))
        out[f"mu0_{group}"] = items
    return out

import bisect
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from twe.contracts import WorldContext, WorldTarget
from twe.data.manifest import Manifest
from twe.data.shards import ShardReader
from twe.preprocess.bspline_targets import BSplineTargets
from twe.preprocess.screen_space import convert


@dataclass(frozen=True)
class QuerySampling:
    slots: int = 64
    min_points: int = 4
    randomize: bool = True
    drop_all_history: float = 0.2
    drop_point_history: float = 0.3
    moving_fraction: float = 0.5
    text_dropout: float = 0.1
    caption_probability: float = 0.8


@dataclass(frozen=True)
class TaskSelection:
    exclude: frozenset = field(default_factory=frozenset)
    only: frozenset | None = None
    shard_fraction: float = 1.0

    def allows(self, task: str | None) -> bool:
        return task not in self.exclude and (self.only is None or task in self.only)

    def shards(self, entries: list) -> list:
        if self.shard_fraction >= 1.0:
            return entries
        rank = lambda e: hashlib.sha1(e.path.encode()).hexdigest()
        keep = {e.path for e in sorted(entries, key=rank)[: math.ceil(len(entries) * self.shard_fraction)]}
        return [e for e in entries if e.path in keep]


def moving_rows(reader: ShardReader) -> np.ndarray:
    return np.asarray(reader.arrays["trace_moving"]) & np.asarray(reader.arrays["anchor_mask"])


class WorldWindowDataset(Dataset):
    def __init__(self, root: Path, split: str, sigma, fitter: BSplineTargets, sampling: QuerySampling,
                 space: str = "screen", tasks: TaskSelection = TaskSelection()):
        self.root = Path(root)
        self.space = space
        self.manifest = Manifest.read(self.root)
        self.readers = [ShardReader(self.root, entry) for entry in tasks.shards(self.manifest.select(split))]
        self.sampling = sampling
        self.usable = [np.flatnonzero((moving_rows(r).sum(-1) >= sampling.min_points)
                                      & np.array([tasks.allows(m.get("task")) for m in r.metas], dtype=bool))
                       for r in self.readers]
        self.offsets = np.cumsum([0] + [len(u) for u in self.usable]).tolist()
        self.sigma = torch.as_tensor(sigma, dtype=torch.float32)
        self.fitter = fitter

    def __len__(self) -> int:
        return self.offsets[-1]

    def locate(self, index: int) -> tuple[ShardReader, int]:
        shard = bisect.bisect_right(self.offsets, index) - 1
        return self.readers[shard], int(self.usable[shard][index - self.offsets[shard]])

    def meta(self, index: int) -> dict:
        reader, row = self.locate(index)
        return {**reader.metas[row], "pool": reader.entry.pool, "split": reader.entry.split}

    def choose(self, index: int, moving: np.ndarray, static: np.ndarray) -> np.ndarray:
        slots, fraction = self.sampling.slots, self.sampling.moving_fraction
        if self.sampling.randomize:
            total = int(torch.randint(self.sampling.min_points, slots + 1, ()))
            order = lambda n: torch.randperm(n).numpy()
        else:
            total = slots
            rng = np.random.default_rng(index)
            order = rng.permutation
        moving_count = min(len(moving), max(self.sampling.min_points, round(total * fraction)))
        static_count = min(len(static), total - moving_count)
        return np.concatenate([moving[order(len(moving))[:moving_count]], static[order(len(static))[:static_count]]])

    def instruction(self, meta: dict) -> str | None:
        caption, original = meta.get("motion_caption"), meta.get("original_instruction")
        if not self.sampling.randomize:
            return caption or original
        if float(torch.rand(())) < self.sampling.text_dropout:
            return None
        return caption if caption and float(torch.rand(())) < self.sampling.caption_probability else original

    def keep_history(self, count: int) -> torch.Tensor:
        if not self.sampling.randomize:
            return torch.ones(count, dtype=torch.bool)
        if float(torch.rand(())) < self.sampling.drop_all_history:
            return torch.zeros(count, dtype=torch.bool)
        return torch.rand(count) >= self.sampling.drop_point_history

    def __getitem__(self, index: int) -> dict:
        reader, row = self.locate(index)
        moving = moving_rows(reader)[row]
        tracked = np.asarray(reader.arrays["anchor_mask"][row]) & np.asarray(reader.arrays["trace_valid"][row]).any(-1)
        rows = self.choose(index, np.flatnonzero(moving), np.flatnonzero(tracked & ~moving))
        slots, count = self.sampling.slots, len(rows)
        pick = lambda key: np.array(reader.arrays[key][row])[rows]
        intrinsics = np.array(reader.arrays["intrinsics"][row])
        size = reader.arrays["rgb"].shape[1]
        xyz = pick("anchor_xyz")
        trace, valid = convert(xyz, pick("trace"), pick("trace_valid"), intrinsics, size, self.space)
        history, seen = convert(xyz, pick("history"), pick("history_valid"), intrinsics, size, self.space)
        seen = seen & self.keep_history(count).numpy()[:, None]
        valid, seen = pad(valid, slots), pad(seen, slots)
        trace = torch.where(valid[..., None], pad(trace, slots) / self.sigma, 0.0)
        history = torch.where(seen[..., None], pad(history, slots) / self.sigma, 0.0)
        weight = pad(pick("trace_reliability"), slots) * valid
        controls, fit_valid = self.fitter.fit(trace, weight)
        mask = pad(np.ones(count, dtype=bool), slots)
        return {
            "rgb": torch.from_numpy(np.array(reader.arrays["rgb"][row])),
            "image_valid": torch.from_numpy(np.array(reader.arrays["image_valid"][row])),
            "anchor_uv": pad(pick("anchor_uv"), slots).float(), "anchor_mask": mask,
            "instruction": self.instruction(reader.metas[row]),
            "trace": trace, "trace_valid": valid, "controls": controls.float(), "fit_valid": fit_valid & mask,
            "moving": pad(moving[rows], slots), "anchor_xyz": pad(xyz, slots).float(),
            "history": history, "history_valid": seen, "intrinsics": torch.from_numpy(intrinsics).float(),
        }


def pad(values: np.ndarray, slots: int) -> torch.Tensor:
    out = torch.zeros((slots, *values.shape[1:]), dtype=torch.from_numpy(values[:0]).dtype)
    out[: len(values)] = torch.from_numpy(np.ascontiguousarray(values))
    return out


def collate(items: list[dict]) -> tuple[WorldContext, WorldTarget]:
    stack = lambda key: torch.stack([item[key] for item in items])
    rgb = stack("rgb").permute(0, 3, 1, 2).float() / 255.0
    context = WorldContext(rgb, stack("image_valid"), stack("anchor_uv"), stack("anchor_mask"),
                           [item["instruction"] for item in items], stack("history"), stack("history_valid"))
    target = WorldTarget(stack("controls"), stack("fit_valid"), stack("trace"), stack("trace_valid"), stack("moving"))
    return context, target

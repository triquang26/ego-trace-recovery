import bisect
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from twe.contracts import WorldContext, WorldTarget
from twe.data.manifest import Manifest
from twe.data.shards import ShardReader
from twe.preprocess.bspline_targets import BSplineTargets


@dataclass(frozen=True)
class QuerySampling:
    slots: int = 64
    min_points: int = 4
    randomize: bool = True
    drop_all_history: float = 0.2
    drop_point_history: float = 0.3
    moving_fraction: float = 0.5
    text_dropout: float = 0.1


def moving_rows(reader: ShardReader) -> np.ndarray:
    return np.asarray(reader.arrays["trace_moving"]) & np.asarray(reader.arrays["anchor_mask"])


class WorldWindowDataset(Dataset):
    def __init__(self, root: Path, split: str, sigma, fitter: BSplineTargets, sampling: QuerySampling):
        self.root = Path(root)
        self.manifest = Manifest.read(self.root)
        self.readers = [ShardReader(self.root, entry) for entry in self.manifest.select(split)]
        self.sampling = sampling
        self.usable = [np.flatnonzero(moving_rows(r).sum(-1) >= sampling.min_points) for r in self.readers]
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

    def instruction(self, text: str | None) -> str | None:
        if self.sampling.randomize and float(torch.rand(())) < self.sampling.text_dropout:
            return None
        return text

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
        pick = lambda key: torch.from_numpy(np.array(reader.arrays[key][row])[rows])
        mask = torch.zeros(slots, dtype=torch.bool)
        mask[:count] = True
        valid = torch.zeros(slots, *reader.arrays["trace_valid"].shape[2:], dtype=torch.bool)
        valid[:count] = pick("trace_valid")
        raw = torch.zeros(slots, *reader.arrays["trace"].shape[2:])
        raw[:count] = pick("trace")
        trace = torch.where(valid[..., None], raw / self.sigma, torch.zeros_like(raw))
        weight = torch.zeros(valid.shape)
        weight[:count] = pick("trace_reliability")
        controls, fit_valid = self.fitter.fit(trace, weight * valid)
        uv = torch.zeros(slots, 2)
        uv[:count] = pick("anchor_uv").float()
        labels = torch.zeros(slots, dtype=torch.bool)
        labels[:count] = torch.from_numpy(moving[rows])
        xyz = torch.zeros(slots, 3)
        xyz[:count] = pick("anchor_xyz").float()
        history_valid = torch.zeros(slots, *reader.arrays["history_valid"].shape[2:], dtype=torch.bool)
        history_valid[:count] = pick("history_valid") & self.keep_history(count)[:, None]
        history = torch.zeros(slots, *reader.arrays["history"].shape[2:])
        history[:count] = pick("history")
        history = torch.where(history_valid[..., None], history / self.sigma, torch.zeros_like(history))
        return {
            "rgb": torch.from_numpy(np.array(reader.arrays["rgb"][row])),
            "image_valid": torch.from_numpy(np.array(reader.arrays["image_valid"][row])),
            "anchor_uv": uv, "anchor_mask": mask,
            "instruction": self.instruction(reader.metas[row].get("original_instruction")),
            "trace": trace, "trace_valid": valid, "controls": controls.float(), "fit_valid": fit_valid & mask,
            "moving": labels, "anchor_xyz": xyz, "history": history, "history_valid": history_valid,
            "intrinsics": torch.from_numpy(np.array(reader.arrays["intrinsics"][row])).float(),
        }


def collate(items: list[dict]) -> tuple[WorldContext, WorldTarget]:
    stack = lambda key: torch.stack([item[key] for item in items])
    rgb = stack("rgb").permute(0, 3, 1, 2).float() / 255.0
    context = WorldContext(rgb, stack("image_valid"), stack("anchor_uv"), stack("anchor_mask"),
                           [item["instruction"] for item in items], stack("history"), stack("history_valid"))
    target = WorldTarget(stack("controls"), stack("fit_valid"), stack("trace"), stack("trace_valid"), stack("moving"))
    return context, target

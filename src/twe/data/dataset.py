import bisect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from twe.contracts import WorldContext, WorldTarget
from twe.data.manifest import Manifest
from twe.data.shards import ShardReader
from twe.preprocess.bspline_targets import BSplineTargets


class WorldWindowDataset(Dataset):
    def __init__(self, root: Path, split: str, sigma, fitter: BSplineTargets):
        self.root = Path(root)
        self.manifest = Manifest.read(self.root)
        self.readers = [ShardReader(self.root, entry) for entry in self.manifest.select(split)]
        self.offsets = np.cumsum([0] + [len(r) for r in self.readers]).tolist()
        self.sigma = torch.as_tensor(sigma, dtype=torch.float32)
        self.fitter = fitter

    def __len__(self) -> int:
        return self.offsets[-1]

    def locate(self, index: int) -> tuple[ShardReader, int]:
        shard = bisect.bisect_right(self.offsets, index) - 1
        return self.readers[shard], index - self.offsets[shard]

    def meta(self, index: int) -> dict:
        reader, row = self.locate(index)
        return {**reader.metas[row], "pool": reader.entry.pool, "split": reader.entry.split}

    def __getitem__(self, index: int) -> dict:
        reader, row = self.locate(index)
        arrays = {key: torch.from_numpy(np.array(value[row])) for key, value in reader.arrays.items()}
        valid = arrays["trace_valid"] & arrays["anchor_mask"][:, None]
        trace = torch.where(valid[..., None], arrays["trace"] / self.sigma, torch.zeros_like(arrays["trace"]))
        weight = arrays["trace_reliability"] * valid
        controls, fit_valid = self.fitter.fit(trace, weight)
        return {
            "rgb": arrays["rgb"], "image_valid": arrays["image_valid"], "anchor_uv": arrays["anchor_uv"].float(),
            "anchor_mask": arrays["anchor_mask"], "instruction": reader.metas[row].get("original_instruction"),
            "trace": trace, "trace_valid": valid, "controls": controls.float(), "fit_valid": fit_valid,
            "moving": arrays["trace_moving"] & arrays["anchor_mask"],
        }


def collate(items: list[dict]) -> tuple[WorldContext, WorldTarget]:
    stack = lambda key: torch.stack([item[key] for item in items])
    rgb = stack("rgb").permute(0, 3, 1, 2).float() / 255.0
    context = WorldContext(rgb, stack("image_valid"), stack("anchor_uv"), stack("anchor_mask"),
                           [item["instruction"] for item in items])
    target = WorldTarget(stack("controls"), stack("fit_valid"), stack("trace"), stack("trace_valid"), stack("moving"))
    return context, target

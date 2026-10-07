import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from twe.data.manifest import ShardEntry

ARRAY_KEYS = ("rgb", "anchor_xyz", "intrinsics", "image_valid", "anchor_uv", "anchor_mask", "trace", "trace_valid", "trace_reliability")


class ShardWriter:
    def __init__(self, root: Path, name: str, pool: str, split: str):
        self.root = Path(root)
        self.name = name
        self.pool = pool
        self.split = split
        self.rows: dict[str, list[np.ndarray]] = {key: [] for key in ARRAY_KEYS}
        self.metas: list[dict] = []

    def __len__(self) -> int:
        return len(self.metas)

    def add(self, arrays: dict, meta: dict) -> None:
        for key in ARRAY_KEYS:
            self.rows[key].append(np.asarray(arrays[key]))
        self.metas.append(meta)

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return {key: np.stack(values) for key, values in self.rows.items()}

    def close(self) -> ShardEntry:
        directory = self.root / self.name
        directory.mkdir(parents=True, exist_ok=True)
        for key, value in self.arrays.items():
            np.save(directory / f"{key}.npy", value)
        with open(directory / "meta.jsonl", "w") as handle:
            for meta in self.metas:
                handle.write(json.dumps(meta) + "\n")
        entry = ShardEntry(self.name, self.pool, self.split, len(self))
        (directory / "entry.json").write_text(json.dumps(asdict(entry)))
        return entry


class ShardReader:
    def __init__(self, root: Path, entry: ShardEntry):
        directory = Path(root) / entry.path
        self.entry = entry
        self.arrays = {key: np.load(directory / f"{key}.npy", mmap_mode="r") for key in ARRAY_KEYS}
        with open(directory / "meta.jsonl") as handle:
            self.metas = [json.loads(line) for line in handle]
        if len(self.metas) != entry.count or len(self.arrays["trace"]) != entry.count:
            raise ValueError(f"shard {entry.path} does not match manifest count {entry.count}")

    def __len__(self) -> int:
        return self.entry.count

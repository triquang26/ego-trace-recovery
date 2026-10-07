import hashlib
import json
from pathlib import Path

import numpy as np


def compute_sigma(traces: list[np.ndarray], valids: list[np.ndarray]) -> np.ndarray:
    total = np.zeros(3)
    count = 0
    for trace, valid in zip(traces, valids):
        selected = trace[valid]
        total += (selected.astype(np.float64) ** 2).sum(0)
        count += selected.shape[0]
    if count == 0:
        raise ValueError("no valid training displacements")
    return np.sqrt(total / count).clip(min=1e-6)


def save_normalizer(path: Path, sigma: np.ndarray, source: str) -> dict:
    values = [float(v) for v in sigma]
    revision = hashlib.sha1(json.dumps([values, source]).encode()).hexdigest()[:12]
    record = {"sigma": values, "source": source, "revision": revision}
    path.write_text(json.dumps(record, indent=2))
    return record


def load_normalizer(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def normalizer_from_dataset(root: Path) -> dict:
    from twe.data.manifest import Manifest
    from twe.data.shards import ShardReader

    manifest = Manifest.read(root)
    readers = [ShardReader(root, entry) for entry in manifest.select("train")]
    traces = [r.arrays["trace"] for r in readers]
    valids = [r.arrays["trace_valid"] for r in readers]
    source = f"{manifest.teacher_revision}:" + ",".join(e.path for e in manifest.select("train"))
    return save_normalizer(root / "normalizer.json", compute_sigma(traces, valids), source)


if __name__ == "__main__":
    import sys

    print(json.dumps(normalizer_from_dataset(Path(sys.argv[1])), indent=2))

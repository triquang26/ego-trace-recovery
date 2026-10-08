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


def save_normalizer(path: Path, sigmas: dict[str, np.ndarray], source: str) -> dict:
    values = {f"sigma_{space}": [float(v) for v in sigma] for space, sigma in sigmas.items()}
    revision = hashlib.sha1(json.dumps([values, source]).encode()).hexdigest()[:12]
    record = {**values, "source": source, "revision": revision}
    path.write_text(json.dumps(record, indent=2))
    return record


def load_normalizer(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def window_targets(reader, space: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    from twe.preprocess.screen_space import convert

    traces, valids = [], []
    size = reader.arrays["rgb"].shape[1]
    for w in range(len(reader)):
        moving = np.asarray(reader.arrays["trace_moving"][w])
        trace, valid = convert(np.asarray(reader.arrays["anchor_xyz"][w]), np.asarray(reader.arrays["trace"][w]),
                               np.asarray(reader.arrays["trace_valid"][w]), np.asarray(reader.arrays["intrinsics"][w]),
                               size, space)
        traces.append(trace)
        valids.append(valid & moving[:, None])
    return traces, valids


def normalizer_from_dataset(root: Path) -> dict:
    from twe.data.manifest import Manifest
    from twe.data.shards import ShardReader
    from twe.preprocess.screen_space import TARGET_SPACES

    manifest = Manifest.read(root)
    readers = [ShardReader(root, entry) for entry in manifest.select("train")]
    sigmas = {}
    for space in TARGET_SPACES:
        pairs = [window_targets(r, space) for r in readers]
        sigmas[space] = compute_sigma([t for p in pairs for t in p[0]], [v for p in pairs for v in p[1]])
    source = f"{manifest.teacher_revision}:" + ",".join(e.path for e in manifest.select("train"))
    return save_normalizer(root / "normalizer.json", sigmas, source)


if __name__ == "__main__":
    import sys

    print(json.dumps(normalizer_from_dataset(Path(sys.argv[1])), indent=2))

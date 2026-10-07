import math
from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class MixtureBatchSampler(Sampler[list[int]]):
    def __init__(self, metas: list[dict], mixture: dict[str, float], batch_size: int, max_null_fraction: float,
                 num_batches: int, seed: int):
        self.recordings: dict[str, dict[bool, dict[str, list[int]]]] = {}
        for index, meta in enumerate(metas):
            pool = self.recordings.setdefault(meta["pool"], {True: defaultdict(list), False: defaultdict(list)})
            key = f"{meta['source']}/{meta['recording_id']}"
            pool[bool(meta.get("original_instruction"))][key].append(index)
        weights = {pool: float(mixture.get(pool, 0.0)) for pool in self.recordings}
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("mixture assigns no weight to available pools")
        self.pools = [pool for pool in weights if weights[pool] > 0]
        self.probs = np.array([weights[pool] for pool in self.pools]) / total
        self.batch_size = batch_size
        self.max_null = math.floor(max_null_fraction * batch_size + 1e-9)
        self.num_batches = num_batches
        self.seed = seed
        self.null_indices = {i for groups in self.recordings.values() for ws in groups[False].values() for i in ws}

    def realized_mixture(self) -> dict[str, float]:
        return {pool: float(p) for pool, p in zip(self.pools, self.probs)}

    def __len__(self) -> int:
        return self.num_batches

    def _draw(self, rng: np.random.Generator, pool: str, allow_null: bool) -> int | None:
        groups = self.recordings[pool]
        candidates = [(flag, key) for flag in (True, False) if flag or allow_null for key in groups[flag]]
        if not candidates:
            return None
        flag, key = candidates[rng.integers(len(candidates))]
        windows = groups[flag][key]
        return windows[rng.integers(len(windows))]

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        has_text = [p for p in self.pools if self.recordings[p][True]]
        for _ in range(self.num_batches):
            batch, nulls = [], 0
            for pool in rng.choice(self.pools, size=self.batch_size, p=self.probs):
                index = self._draw(rng, pool, nulls < self.max_null)
                if index is None:
                    index = self._draw(rng, has_text[rng.integers(len(has_text))], False)
                nulls += int(index in self.null_indices)
                batch.append(index)
            yield batch


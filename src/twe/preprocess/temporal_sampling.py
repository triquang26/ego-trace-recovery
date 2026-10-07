from dataclasses import dataclass

import numpy as np


@dataclass
class SamplePlan:
    lower: np.ndarray
    upper: np.ndarray
    weight: np.ndarray
    valid: np.ndarray


def plan_future_samples(
    timestamps: np.ndarray, current_time: float, offsets: list[float], max_gap: float, tolerance: float
) -> SamplePlan:
    targets = current_time + np.asarray(offsets, dtype=np.float64)
    count = len(timestamps)
    upper = np.searchsorted(timestamps, targets, side="left").clip(0, count - 1)
    lower = (upper - 1).clip(0, count - 1)
    t_lo = timestamps[lower]
    t_hi = timestamps[upper]
    exact = np.abs(t_hi - targets) <= tolerance
    bracketed = (t_lo <= targets) & (targets <= t_hi) & (t_hi - t_lo <= max_gap) & (t_hi > t_lo)
    span = np.where(t_hi > t_lo, t_hi - t_lo, 1.0)
    weight = np.where(exact, 1.0, np.clip((targets - t_lo) / span, 0.0, 1.0))
    valid = exact | bracketed
    return SamplePlan(lower, upper, weight, valid)


def gather_tracks(
    points: np.ndarray, reliability: np.ndarray, plan: SamplePlan
) -> tuple[np.ndarray, np.ndarray]:
    w = plan.weight[None, :, None]
    lo = points[:, plan.lower]
    hi = points[:, plan.upper]
    gathered = np.where(w >= 1.0, hi, (1 - w) * lo + w * hi)
    rel_lo = reliability[:, plan.lower]
    rel_hi = reliability[:, plan.upper]
    rel = np.where(plan.weight[None] >= 1.0, rel_hi, np.minimum(rel_lo, rel_hi))
    gathered = np.where(plan.valid[None, :, None], gathered, np.nan)
    rel = np.where(plan.valid[None], rel, 0.0)
    return gathered, rel

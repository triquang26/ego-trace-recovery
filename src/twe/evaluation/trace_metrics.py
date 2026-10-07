import torch
from torch import Tensor

from twe.contracts import WorldTarget
from twe.preprocess.bspline_targets import BSplineTargets


def trace_errors(pred: Tensor, target: WorldTarget, anchor_mask: Tensor, dynamic_threshold: float) -> dict[str, Tensor]:
    valid = target.trace_valid & anchor_mask[..., None]
    distance = torch.linalg.norm(pred.float() - target.trace.float(), dim=-1)
    motion = torch.where(valid, torch.linalg.norm(target.trace.float(), dim=-1), torch.zeros_like(distance))
    dynamic = motion.amax(-1) > dynamic_threshold
    last = valid[..., -1]
    stats = {
        "ade_sum": (distance * valid).sum(), "ade_count": valid.sum(),
        "fde_sum": (distance[..., -1] * last).sum(), "fde_count": last.sum(),
        "dyn_ade_sum": (distance * valid * dynamic[..., None]).sum(),
        "dyn_ade_count": (valid & dynamic[..., None]).sum(),
        "zero_ade_sum": (motion * valid).sum(),
    }
    return {k: v.detach().double().cpu() for k, v in stats.items()}


def reconstruction_error(fitter: BSplineTargets, target: WorldTarget, anchor_mask: Tensor) -> dict[str, Tensor]:
    decoded = fitter.decode(target.controls.double()).float()
    valid = target.trace_valid & (anchor_mask & target.fit_valid)[..., None]
    distance = torch.linalg.norm(decoded - target.trace.float(), dim=-1)
    return {"fit_sum": (distance * valid).sum().double().cpu(), "fit_count": valid.sum().double().cpu()}


def validity_calibration(logits: Tensor, target: Tensor, mask: Tensor, bins: int = 10) -> dict[str, Tensor]:
    prob = torch.sigmoid(logits.float())[mask]
    label = target.float()[mask]
    index = (prob * bins).long().clamp(max=bins - 1)
    out = {}
    for name, values in (("count", torch.ones_like(prob)), ("prob", prob), ("label", label)):
        out[f"ece_{name}"] = torch.zeros(bins, dtype=torch.float64).index_add_(0, index.cpu(), values.double().cpu())
    return out


def summarize(totals: dict[str, Tensor]) -> dict[str, float]:
    ratio = lambda a, b: float(totals[a] / totals[b].clamp_min(1))
    count = totals["ece_count"]
    ece = (totals["ece_prob"] - totals["ece_label"]).abs().sum() / count.sum().clamp_min(1)
    return {
        "ade": ratio("ade_sum", "ade_count"),
        "fde": ratio("fde_sum", "fde_count"),
        "dynamic_ade": ratio("dyn_ade_sum", "dyn_ade_count"),
        "zero_motion_ade": ratio("zero_ade_sum", "ade_count"),
        "fit_error": ratio("fit_sum", "fit_count"),
        "validity_ece": float(ece),
    }

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from twe.data.dataset import QuerySampling, TaskSelection, WorldWindowDataset, collate
from twe.data.traceextract import traceextract_items
from twe.evaluation.demo import auroc

METRICS = ("ade", "ade_mean5", "minade5", "zero", "ade_nohist")


def point_ade(pred: torch.Tensor, trace: torch.Tensor, valid: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    distance = torch.linalg.norm((pred - trace)[..., :2] * scale[:2], dim=-1) * 100
    return (distance * valid).sum(-1) / valid.sum(-1).clamp_min(1)


class BenchmarkRunner:
    def __init__(self, module, fitter, sigma, device, samples: int = 5, steps: int = 4, batch: int = 32):
        self.module, self.fitter = module.eval(), fitter
        self.scale = torch.as_tensor(sigma, dtype=torch.float32, device=device)
        self.device, self.samples, self.steps, self.batch = torch.device(device), samples, steps, batch

    @torch.no_grad()
    def decode(self, inputs, seed: int) -> torch.Tensor:
        controls = self.module.sample_controls(inputs, self.steps, torch.Generator().manual_seed(seed))
        return self.fitter.decode(controls.double()).float()

    @torch.no_grad()
    def batch_stats(self, items: list[dict], sums: dict, scores: list) -> None:
        context, target = collate(items)
        context, target = context.to(self.device), target.to(self.device)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            inputs = self.module.encode(context)
            preds = torch.stack([self.decode(inputs, seed) for seed in range(self.samples)])
            blind = self.decode(self.module.encode(context.without_history()), 0)
            batch = context.anchor_uv.shape[0]
            noise = self.module.guidance_noise(batch, self.device)
            logits = self.module(inputs, noise, torch.ones(batch, device=self.device)).motion_logits.float()
        valid = (target.trace_valid & (context.anchor_mask & target.moving)[..., None]).float()
        per_sample = torch.stack([point_ade(p, target.trace, valid, self.scale) for p in preds])
        keep = valid.sum(-1) > 0
        zeros = torch.zeros_like(target.trace)
        values = {"ade": per_sample[0], "ade_mean5": point_ade(preds.mean(0), target.trace, valid, self.scale),
                  "minade5": per_sample.min(0).values, "zero": point_ade(zeros, target.trace, valid, self.scale),
                  "ade_nohist": point_ade(blind, target.trace, valid, self.scale)}
        sums["points"] += float(keep.sum())
        for key, value in values.items():
            sums[key] += float(value[keep].sum())
        mask = context.anchor_mask
        scores.append((torch.sigmoid(logits)[mask].cpu().numpy(), target.moving[mask].cpu().numpy()))

    def run(self, groups: dict[str, list[dict]]) -> dict[str, dict]:
        report = {}
        for name, items in groups.items():
            if not items:
                continue
            sums, scores = defaultdict(float), []
            for start in range(0, len(items), self.batch):
                self.batch_stats(items[start : start + self.batch], sums, scores)
            count = max(sums["points"], 1.0)
            probs = np.concatenate([s for s, _ in scores])
            labels = np.concatenate([l for _, l in scores]).astype(bool)
            report[name] = {key: round(sums[key] / count, 4) for key in METRICS}
            report[name].update({"windows": len(items), "moving_points": int(sums["points"]),
                                 "motion_auroc": round(auroc(probs, labels), 4) if (~labels).any() else None})
        return report


def egodex_groups(root: Path, cfg, fitter, sigma, heldout: frozenset, limit: int = 600, seed: int = 0) -> dict:
    sampling = QuerySampling(cfg.num_anchors, 4, False)
    seen = WorldWindowDataset(root, "validation", sigma, fitter, sampling, cfg.target_space, TaskSelection(heldout))
    groups = {"egodex_val_seen_tasks": [seen[i] for i in range(min(len(seen), limit))]}
    if heldout:
        held = [WorldWindowDataset(root, split, sigma, fitter, sampling, cfg.target_space,
                                   TaskSelection(frozenset(), heldout)) for split in ("train", "validation")]
        rng = np.random.default_rng(seed)
        pool = [(d, i) for d in held for i in range(len(d))]
        picks = rng.permutation(len(pool))[:limit]
        groups["egodex_heldout_tasks"] = [pool[i][0][pool[i][1]] for i in sorted(picks)]
    return groups


def benchmark_groups(data_root: Path, traceextract_root: Path | None, cfg, fitter, sigma,
                     heldout: frozenset) -> dict[str, list[dict]]:
    groups = egodex_groups(data_root, cfg, fitter, sigma, heldout)
    if traceextract_root is not None and Path(traceextract_root).exists():
        groups.update(traceextract_items(traceextract_root, cfg, fitter, sigma))
    return groups

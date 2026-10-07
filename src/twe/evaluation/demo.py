import json
from pathlib import Path

import numpy as np
import torch

from twe.data.dataset import collate
from twe.evaluation.demo_render import render_case

@torch.no_grad()
def predict(module, fitter, context, seed: int, steps: int) -> np.ndarray:
    inputs = module.encode(context)
    controls = module.sample_controls(inputs, steps, torch.Generator().manual_seed(seed))
    return fitter.decode(controls.double()).float().cpu().numpy()


def sample_stats(pred, trace, valid, mask, moving) -> dict:
    v = valid & mask[:, None]
    dist = np.linalg.norm(pred - trace, axis=-1)
    motion = np.linalg.norm(trace, axis=-1) * v
    dynamic = moving & mask
    count = max(v.sum(), 1)
    dyn = v & dynamic[:, None]
    return {"ade": float((dist * v).sum() / count), "zero_ade": float(motion.sum() / count),
            "dynamic_ade": float((dist * dyn).sum() / max(dyn.sum(), 1)), "dynamic_points": int(dynamic.sum()),
            "pred_motion": float((np.linalg.norm(pred, axis=-1) * v).sum() / count)}


def min_ade(preds: np.ndarray, trace: np.ndarray, valid: np.ndarray) -> float:
    dist = np.linalg.norm(preds - trace[None], axis=-1) * valid[None]
    per_point = dist.sum(-1) / np.maximum(valid.sum(-1), 1)[None]
    keep = valid.any(-1)
    return float(per_point.min(0)[keep].mean()) if keep.any() else 0.0


class DemoBuilder:
    def __init__(self, module, fitter, dataset, sigma, device, steps: int = 4):
        self.module, self.fitter, self.dataset = module.eval(), fitter, dataset
        self.sigma = np.asarray(sigma, dtype=np.float32)
        self.device, self.steps = device, steps

    def context(self, index: int, instruction=...):
        item = dict(self.dataset[index])
        if instruction is not ...:
            item["instruction"] = instruction
        context, target = collate([item])
        return context.to(self.device), target, item

    def scan(self, indices: list[int]) -> list[dict]:
        rows = []
        for index in indices:
            context, target, item = self.context(index)
            preds = np.stack([predict(self.module, self.fitter, context, seed, self.steps)[0] for seed in range(5)])
            trace, valid = target.trace[0].numpy(), target.trace_valid[0].numpy()
            mask = context.anchor_mask[0].cpu().numpy()
            stats = sample_stats(preds[0], trace, valid, mask, target.moving[0].numpy())
            stats["min_ade"] = min_ade(preds, trace, valid & mask[:, None])
            rows.append({"index": index, "instruction": item["instruction"], **stats})
        return rows

    def render(self, out: Path, name: str, index: int, variants: list[tuple[str, object, int]], teacher: bool):
        _, _, item = self.context(index)
        rows = np.flatnonzero(item["anchor_mask"].numpy())
        valid = item["trace_valid"].numpy()
        panels = [("Teacher", item["trace"].numpy() * self.sigma, valid)] if teacher else []
        preds = []
        for title, instruction, seed in variants:
            context, _, _ = self.context(index, instruction)
            pred = predict(self.module, self.fitter, context, seed, self.steps)[0]
            preds.append(pred)
            panels.append((title, pred * self.sigma, np.ones_like(valid)))
        meta = self.dataset.meta(index)
        render_case(out / f"{name}.png", item["rgb"].numpy(), item["anchor_xyz"].numpy(), item["intrinsics"].numpy(),
                    panels, rows, f"{meta['sample_id']} | {item['instruction'] or 'no instruction'}")
        return preds


def build_demo(builder: DemoBuilder, out: Path, count: int = 200, seed: int = 0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(builder.dataset), min(count, len(builder.dataset)), replace=False).tolist())
    rows = builder.scan(indices)
    with_text = [r for r in rows if r["instruction"] and r["dynamic_points"] > 0]
    ranked = sorted(with_text, key=lambda r: r["dynamic_ade"])
    picks = {"best": ranked[:4], "worst": ranked[-4:][::-1],
             "random": [with_text[i] for i in rng.choice(len(with_text), min(4, len(with_text)), replace=False)]}
    cases = []
    for group, chosen in picks.items():
        for n, row in enumerate(chosen):
            name = f"{group}_{n}"
            builder.render(out, name, row["index"], [("Model", ..., 0)], True)
            cases.append({"image": f"{name}.png", "group": group, **row})
    others = [r["instruction"] for r in with_text]
    for n, row in enumerate(ranked[: len(ranked) // 2][:4]):
        swap = next((t for t in rng.permutation(others) if t != row["instruction"]), None)
        variants = [(f"original: {row['instruction']}", ..., 0), (f"swapped: {swap}", swap, 0),
                    ("no instruction", None, 0)]
        preds = builder.render(out, f"swap_{n}", row["index"], variants, False)
        change = [float(np.linalg.norm(p - preds[0], axis=-1).mean()) for p in preds[1:]]
        cases.append({"image": f"swap_{n}.png", "group": "instruction", **row, "swap": swap,
                      "mean_change_swap": change[0], "mean_change_null": change[1]})
    for n, row in enumerate(ranked[:2]):
        builder.render(out, f"seeds_{n}", row["index"], [(f"seed {s}", ..., s) for s in range(3)], False)
        cases.append({"image": f"seeds_{n}.png", "group": "seeds", **row})
    keys = ("ade", "min_ade", "zero_ade", "dynamic_ade", "pred_motion")
    summary = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    record = {"summary": summary, "scanned": len(rows), "cases": cases}
    (out / "demo.json").write_text(json.dumps(record, indent=2))
    return record

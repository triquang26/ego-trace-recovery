import json
from pathlib import Path

import numpy as np
import torch

from twe.data.dataset import collate
from twe.evaluation.demo_render import project, render_case

PRED, TEACHER = "#ff3b30", "#34c759"


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
            pred = predict(self.module, self.fitter, context, 0, self.steps)[0]
            stats = sample_stats(pred, target.trace[0].numpy(), target.trace_valid[0].numpy(),
                                 context.anchor_mask[0].cpu().numpy(), target.moving[0].numpy())
            rows.append({"index": index, "instruction": item["instruction"], **stats})
        return rows

    def paths(self, index: int, pred: np.ndarray, with_teacher: bool) -> dict:
        reader, row = self.dataset.locate(index)
        xyz, k = np.array(reader.arrays["anchor_xyz"][row]), np.array(reader.arrays["intrinsics"][row])
        item = self.dataset[index]
        valid = item["trace_valid"].numpy()
        out = {"model": (project(xyz, pred * self.sigma, k), np.ones_like(valid), PRED)}
        if with_teacher:
            out["teacher"] = (project(xyz, item["trace"].numpy() * self.sigma, k), valid, TEACHER)
        return out

    def render(self, out: Path, name: str, index: int, variants: list[tuple[str, object, int]], teacher: bool):
        _, _, item = self.context(index)
        rgb = item["rgb"].numpy()
        mask = item["anchor_mask"].numpy()
        dynamic = item["moving"].numpy()
        panels, preds = [], []
        for title, instruction, seed in variants:
            context, _, _ = self.context(index, instruction)
            pred = predict(self.module, self.fitter, context, seed, self.steps)[0]
            preds.append(pred)
            panels.append({"paths": self.paths(index, pred, teacher), "mask": mask, "title": title,
                           "dynamic": dynamic})
        meta = self.dataset.meta(index)
        render_case(out / f"{name}.png", rgb, panels, f"{meta['sample_id']}")
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
            builder.render(out, name, row["index"], [(f"model vs teacher | {row['instruction']}", ..., 0)], True)
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
    summary = {key: float(np.mean([r[key] for r in rows])) for key in ("ade", "zero_ade", "dynamic_ade", "pred_motion")}
    record = {"summary": summary, "scanned": len(rows), "cases": cases}
    (out / "demo.json").write_text(json.dumps(record, indent=2))
    return record

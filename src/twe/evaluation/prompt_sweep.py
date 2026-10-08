import json
from pathlib import Path

import numpy as np
import torch

from twe.evaluation.demo import DemoBuilder
from twe.evaluation.demo_render import render_case

PROBE_PROMPTS = (("move left", "move both hands to the left side of the table."),
                 ("lift up", "lift the object up toward the camera."),
                 ("stay still", "keep both hands still on the table."),
                 ("no prompt", None))


@torch.no_grad()
def guided_prediction(builder: DemoBuilder, context, instruction: str | None, guidance: float,
                      seed: int) -> tuple[np.ndarray, np.ndarray]:
    module = builder.module
    text = module.encode(context.with_instructions([instruction]))
    null = module.encode(context.with_instructions([None]))
    controls = module.sample_controls(text, builder.steps, torch.Generator().manual_seed(seed), guidance, null)
    probs = module.motion_probability(text, guidance, null)
    return builder.fitter.decode(controls.double()).float().cpu().numpy()[0], probs.cpu().numpy()[0]


def prompt_sweep(builder: DemoBuilder, index: int, prompts, guidance: float, path: Path, seed: int = 0) -> dict:
    context, _, item = builder.context(index)
    rows = np.flatnonzero(item["anchor_mask"].numpy())
    valid = item["trace_valid"].numpy()
    original = item["instruction"]
    panels = [("teacher", builder.displacement(item, item["trace"].numpy()), valid)]
    predictions = []
    for label, text in (("original", original), *prompts):
        pred, probs = guided_prediction(builder, context, text, guidance, seed)
        predictions.append(builder.displacement(item, pred))
        panels.append((f"{label} (w={guidance:g})", predictions[-1], np.ones_like(valid), probs))
    meta = builder.dataset.meta(index)
    render_case(path, item["rgb"].numpy(), item["anchor_xyz"].numpy(), item["intrinsics"].numpy(), panels, rows,
                f"{meta['sample_id']} | {original or 'no instruction'}")
    shift = [float(np.linalg.norm(p - predictions[0], axis=-1)[rows].mean()) for p in predictions[1:]]
    return {"index": index, "image": Path(path).name, "guidance": guidance, "original": original,
            "prompts": [text for _, text in prompts], "shift_vs_original": shift}


def most_dynamic(builder: DemoBuilder, count: int) -> list[int]:
    moving = [int(builder.dataset[i]["moving"].sum()) for i in range(len(builder.dataset))]
    tasks, chosen = set(), []
    for index in sorted(range(len(moving)), key=lambda i: -moving[i]):
        meta = builder.dataset.meta(index)
        task = meta.get("task", meta.get("recording_id"))
        if task not in tasks:
            tasks.add(task)
            chosen.append(index)
        if len(chosen) == count:
            break
    return chosen


def build_prompt_sweep(builder: DemoBuilder, out: Path, scenes: int = 3, guidances=(1.0, 4.0)) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    records = [prompt_sweep(builder, index, PROBE_PROMPTS, w, out / f"scene{n}_w{w:g}.png")
               for n, index in enumerate(most_dynamic(builder, scenes)) for w in guidances]
    (out / "prompts.json").write_text(json.dumps(records, indent=2))
    return {"cases": records}

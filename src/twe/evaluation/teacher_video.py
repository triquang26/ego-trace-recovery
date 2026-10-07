import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from twe.data.manifest import ShardEntry
from twe.data.shards import ShardReader
from twe.evaluation.demo_render import project
from twe.preprocess.letterbox import letterbox

PALETTE = [(255, 59, 48), (255, 149, 0), (255, 204, 0), (52, 199, 89), (0, 199, 190), (0, 122, 255), (175, 82, 222)]


def most_dynamic(root: Path, count: int, threshold: float = 0.05) -> list[tuple[ShardReader, int]]:
    entries = [ShardEntry(**json.loads(path.read_text())) for path in sorted(Path(root).glob("*/entry.json"))]
    scored = []
    for entry in entries:
        reader = ShardReader(root, entry)
        motion = np.linalg.norm(np.asarray(reader.arrays["trace"]), axis=-1) * np.asarray(reader.arrays["trace_valid"])
        moving = (motion.max(-1) > threshold).sum(-1)
        for row in range(len(reader)):
            scored.append((int(moving[row]), reader.metas[row].get("task", ""), reader, row))
    scored.sort(key=lambda item: -item[0])
    chosen, tasks = [], set()
    for moving, task, reader, row in scored:
        if task not in tasks:
            chosen.append((reader, row))
            tasks.add(task)
        if len(chosen) == count:
            break
    return chosen


def render_window(reader: ShardReader, row: int, frames: np.ndarray, times: np.ndarray, size: int, path: Path,
                  threshold: float = 0.05, upscale: int = 2) -> dict:
    arrays = {key: np.asarray(value[row]) for key, value in reader.arrays.items()}
    meta = reader.metas[row]
    pixels = project(arrays["anchor_xyz"], arrays["trace"], arrays["intrinsics"]) * upscale
    valid = np.concatenate([np.ones((len(pixels), 1), bool), arrays["trace_valid"]], 1) & arrays["anchor_mask"][:, None]
    motion = np.linalg.norm(arrays["trace"], axis=-1) * arrays["trace_valid"]
    dynamic = motion.max(-1) > threshold
    offsets = np.concatenate([[0.0], meta["future_offsets_seconds"]])
    base = Image.fromarray(arrays["rgb"]).resize((size * upscale, size * upscale))
    images = []
    for frame, time in zip(frames, times):
        left = Image.fromarray(letterbox(frame, size)[0]).resize(base.size)
        right = base.copy()
        draw = ImageDraw.Draw(right)
        upto = int(np.searchsorted(offsets, time, side="right"))
        for i in np.nonzero(arrays["anchor_mask"])[0]:
            keep = [tuple(p) for p, ok in zip(pixels[i, :upto], valid[i, :upto]) if ok]
            color = PALETTE[i % len(PALETTE)] if dynamic[i] else (200, 200, 200)
            if len(keep) > 1:
                draw.line(keep, fill=color, width=3 if dynamic[i] else 1)
            x, y = pixels[i, 0]
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(255, 255, 255))
        canvas = Image.new("RGB", (base.width * 2 + 8, base.height + 28), (20, 20, 20))
        canvas.paste(left, (0, 28))
        canvas.paste(right, (base.width + 8, 28))
        ImageDraw.Draw(canvas).text((6, 6), f"t+{time:.2f}s  {(meta.get('original_instruction') or '')[:90]}",
                                    fill=(255, 255, 255))
        images.append(canvas)
    images[0].save(path, save_all=True, append_images=images[1:], duration=120, loop=0)
    return {"file": path.name, "sample_id": meta["sample_id"], "instruction": meta.get("original_instruction"),
            "dynamic_points": int(dynamic.sum()), "valid_fraction": float(arrays["trace_valid"].mean())}


def write_index(out: Path, records: list[dict]) -> None:
    (out / "index.json").write_text(json.dumps(records, indent=2))

import json
from pathlib import Path

import numpy as np
from PIL import Image

from twe.data.manifest import ShardEntry
from twe.data.shards import ShardReader
from twe.evaluation.demo_render import project, shared_extent
from twe.evaluation.trace_style import absolute_points, draw_history, draw_paths, grayscale, hstack, titled, top_down
from twe.preprocess.letterbox import letterbox


def most_dynamic(root: Path, count: int) -> list[tuple[ShardReader, int]]:
    entries = [ShardEntry(**json.loads(path.read_text())) for path in sorted(Path(root).glob("*/entry.json"))]
    scored = []
    for entry in entries:
        reader = ShardReader(root, entry)
        moving = np.asarray(reader.arrays["trace_moving"]).sum(-1)
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


def render_window(reader: ShardReader, row: int, frames: np.ndarray, times: np.ndarray, path: Path,
                  side: int = 448) -> dict:
    arrays = {key: np.asarray(value[row]) for key, value in reader.arrays.items()}
    meta = reader.metas[row]
    rows = np.flatnonzero(arrays["trace_moving"] & arrays["anchor_mask"])
    size = arrays["rgb"].shape[0]
    upscale = side / size
    pixels = project(arrays["anchor_xyz"], arrays["trace"], arrays["intrinsics"]) * upscale
    points, keep = absolute_points(arrays["anchor_xyz"], arrays["trace"], arrays["trace_valid"])
    extent = shared_extent([(points, keep)], rows)
    offsets = np.concatenate([[0.0], meta["future_offsets_seconds"]])
    base = grayscale(Image.fromarray(arrays["rgb"]).resize((side, side)))
    past = project(arrays["anchor_xyz"], arrays["history"], arrays["intrinsics"])[:, 1:] * upscale
    past_valid = arrays["history_valid"]
    past = np.concatenate([past, pixels[:, :1]], 1)
    past_valid = np.concatenate([past_valid, np.ones((len(past_valid), 1), bool)], 1)
    base = draw_history(base, past, past_valid, rows)
    images = []
    for frame, time in zip(frames, times):
        upto = int(np.searchsorted(offsets, time, side="right"))
        video = Image.fromarray(letterbox(frame, size)[0]).resize((side, side))
        teacher = draw_paths(base.copy(), pixels, keep, rows, upto)
        sheet = hstack([titled(video, f"video t+{time:.2f}s"), titled(teacher, f"teacher, {len(rows)} moving points"),
                        titled(top_down(points, keep, rows, upto, side, extent), "teacher x-z")])
        images.append(titled(sheet, (meta.get("original_instruction") or "")[:150], 26))
    images[0].save(path, save_all=True, append_images=images[1:], duration=120, loop=0)
    return {"file": path.name, "sample_id": meta["sample_id"], "instruction": meta.get("original_instruction"),
            "dynamic_points": int(len(rows)), "valid_fraction": float(arrays["trace_valid"].mean())}


def write_index(out: Path, records: list[dict]) -> None:
    (out / "index.json").write_text(json.dumps(records, indent=2))


def render_dataset(root: Path, raw_root: Path, out: Path, count: int, frame_step: int) -> list[dict]:
    from twe.data.egodex import egodex_recording

    out.mkdir(parents=True, exist_ok=True)
    records = []
    for n, (reader, row) in enumerate(most_dynamic(root, count)):
        meta = reader.metas[row]
        part, task, index = meta["recording_id"].split("/")
        raw = Path(raw_root) / part / task
        recording = egodex_recording(part, task, raw / f"{index}.hdf5", raw / f"{index}.mp4", frame_step)
        now = meta["current_timestamp_seconds"]
        start = int(np.argmin(np.abs(recording.timestamps - now)))
        stop = int(np.searchsorted(recording.timestamps, now + 2.0, side="right"))
        frames = recording.read_frames(np.arange(start, stop))
        times = recording.timestamps[start:stop] - recording.timestamps[start]
        records.append(render_window(reader, row, frames, times, out / f"teacher_{n}.gif"))
    write_index(out, records)
    return records

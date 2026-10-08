import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from twe.data.traceextract import DEFAULT_FPS, TraceExtractEpisode, resample
from twe.evaluation.rollout import EpisodeGeometry, TraceRollout
from twe.evaluation.trace_style import age_color, grayscale, hstack, titled

BUCKETS = ((0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0))


def long_slot(episode: TraceExtractEpisode, frames: int, history: int, max_points: int):
    best, rows_best = None, np.zeros(0, dtype=int)
    for slot in range(len(episode.frames)):
        if episode.frames[slot] < history:
            continue
        lo, hi = int(episode.offsets[slot]), int(episode.offsets[slot + 1])
        ok = np.flatnonzero(np.asarray(episode.future_valid[lo:hi]).sum(1) > frames) + lo
        if len(ok) > len(rows_best) or (len(ok) == len(rows_best) and best is not None and
                                       abs(slot - len(episode.frames) / 2) < abs(best - len(episode.frames) / 2)):
            best, rows_best = slot, ok
    return best, rows_best[np.linspace(0, len(rows_best) - 1, min(max_points, len(rows_best))).astype(int)]


def bucket_errors(pred: np.ndarray, truth: np.ndarray, fps: float, width: int) -> dict:
    error = np.linalg.norm(pred[..., :2] - truth[..., :2], axis=-1) / width * 100
    still = np.linalg.norm(truth[..., :2] - truth[:, :1, :2], axis=-1) / width * 100
    t = np.arange(truth.shape[1]) / fps
    out = {}
    for lo, hi in BUCKETS:
        keep = (t > lo) & (t <= hi)
        if not keep.any():
            continue
        out[f"{lo:g}-{hi:g}s"] = {"ade": float(error[:, keep].mean()), "zero": float(still[:, keep].mean())}
    return out


def draw_tracks(image: np.ndarray, tracks: np.ndarray, title: str, scale: int = 2) -> Image.Image:
    canvas = grayscale(Image.fromarray(image)).resize((image.shape[1] * scale, image.shape[0] * scale))
    draw = ImageDraw.Draw(canvas)
    steps = tracks.shape[1]
    for track in tracks:
        for k in range(1, steps):
            color = age_color(k / (steps - 1))
            draw.line([tuple(track[k - 1, :2] * scale), tuple(track[k, :2] * scale)], fill=color, width=2)
        x, y = track[0, :2] * scale
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(148, 0, 211))
    return titled(canvas, title)


def segment_gif(geometry: EpisodeGeometry, start: int, truth_world: np.ndarray, rollout_world: np.ndarray,
                stride_frames: int, path: Path, scale: int = 2) -> None:
    frames = []
    for k in range(0, rollout_world.shape[1] - 1, stride_frames):
        frame = min(start + k, len(geometry.images) - 1)
        image = grayscale(Image.fromarray(np.asarray(geometry.images[frame]))).resize(
            (geometry.images.shape[2] * scale, geometry.images.shape[1] * scale))
        draw = ImageDraw.Draw(image)
        for source, color in ((truth_world, (235, 235, 235)), (rollout_world, None)):
            pixels = geometry.to_pixels(frame, source[:, k : k + 2 * stride_frames + 1]) * scale
            for track in pixels:
                for j in range(1, track.shape[0]):
                    fill = color or age_color(j / max(track.shape[0] - 1, 1))
                    draw.line([tuple(track[j - 1, :2]), tuple(track[j, :2])], fill=fill, width=2 if color else 3)
        label = f"t = {k / geometry.fps:.1f} s   white: teacher   color: model (next 2 s)"
        draw.text((8, 8), label, fill=(255, 255, 0))
        frames.append(image)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=700, loop=0)


def rollout_episode(rollout: TraceRollout, path: Path, out: Path, horizon: float = 6.0, stride: float = 1.0,
                    max_points: int = 24, fps: float = DEFAULT_FPS) -> dict:
    episode = TraceExtractEpisode(path, fps)
    cameras = np.load(path / "cameras.npz")
    geometry = EpisodeGeometry(episode.images, cameras["intrinsics"].astype(np.float64),
                               cameras["extrinsics"].astype(np.float64), fps)
    frames = int(round(horizon * fps))
    slot, rows = long_slot(episode, frames, int(np.ceil(0.5 * fps)) + 1, max_points)
    if slot is None or len(rows) < 4:
        return {"episode": path.name, "skipped": True}
    start = int(episode.frames[slot])
    truth = np.asarray(episode.future[rows], dtype=np.float64)[:, : frames + 1]
    cfg = rollout.cfg
    times = np.arange(cfg.history_steps, 0, -1) * cfg.horizon_seconds / cfg.future_steps
    past, past_ok = resample(np.asarray(episode.past[rows], dtype=np.float64), np.asarray(episode.past_valid[rows]),
                             times, fps)
    past = past if past_ok.all() else None
    current = truth[:, 0]
    runs = {"replan_observed": (past, True), "closed_loop": (past, True), "closed_loop_no_history": (None, True),
            "open_loop": (past, False)}
    width = episode.images.shape[2]
    truth_world = geometry.to_world(start, truth)
    observed = {"replan_observed": truth_world}
    record = {"episode": path.name, "start_frame": start, "start_seconds": start / fps,
              "episode_seconds": len(episode.images) / fps, "points": int(len(rows)),
              "instruction": episode.instruction, "history_available": past is not None, "errors": {},
              "camera_hw": [int(cameras["height"]), int(cameras["width"])] if "height" in cameras else None,
              "image_hw": list(episode.images.shape[1:3])}
    panels = [draw_tracks(np.asarray(episode.images[start]), truth, f"teacher {horizon:g} s")]
    for name, (history, closed) in runs.items():
        world = rollout.run(geometry, start, current, history, episode.instruction, horizon, stride, closed,
                            observed=observed.get(name))
        pixels = geometry.to_pixels(start, world)
        record["errors"][name] = bucket_errors(pixels, truth, fps, width)
        panels.append(draw_tracks(np.asarray(episode.images[start]), pixels, name.replace("_", " ")))
        if name in ("closed_loop", "replan_observed"):
            segment_gif(geometry, start, truth_world, world, int(round(stride * fps)), out / f"{path.name}_{name}.gif")
    hstack(panels).save(out / f"{path.name}.png")
    return record


def rollout_benchmark(rollout: TraceRollout, episodes: list[Path], out: Path, **kwargs) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = [rollout_episode(rollout, path, out, **kwargs) for path in episodes]
    timing = np.asarray(rollout.seconds[1:] or rollout.seconds)
    report = {"episodes": rows, "segment_ms_median": float(np.median(timing) * 1000) if len(timing) else None}
    (out / "rollout.json").write_text(json.dumps(report, indent=2))
    return report

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from twe.data.traceextract import DEFAULT_FPS, TraceExtractEpisode
from twe.evaluation.trace_style import age_color, grayscale, hstack, titled
from twe.preprocess.teacher import TrackTeacher


def densest_slot(episode: TraceExtractEpisode, steps: int, context: int) -> tuple[int | None, np.ndarray]:
    best, keep_best = None, np.zeros(0, dtype=bool)
    for slot, frame in enumerate(episode.frames):
        if frame < context or frame + steps > len(episode.images):
            continue
        lo, hi = int(episode.offsets[slot]), int(episode.offsets[slot + 1])
        keep = np.asarray(episode.future_valid[lo:hi, :steps]).all(1)
        if keep.sum() > keep_best.sum():
            best, keep_best = slot, keep
    return best, keep_best


def project_tracks(points_world: np.ndarray, world_to_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    camera = np.einsum("ij,ntj->nti", world_to_camera[:3, :3], points_world) + world_to_camera[:3, 3]
    z = camera[..., 2]
    u = intrinsics[0, 0] * camera[..., 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * camera[..., 1] / z + intrinsics[1, 2]
    return np.stack([u, v, z], -1)


def trace_panel(image: np.ndarray, traces: np.ndarray, title: str, scale: int = 2) -> Image.Image:
    canvas = grayscale(Image.fromarray(image)).resize((image.shape[1] * scale, image.shape[0] * scale))
    draw = ImageDraw.Draw(canvas)
    steps = traces.shape[1]
    for trace in traces:
        for k in range(1, steps):
            draw.line([tuple(trace[k - 1, :2] * scale), tuple(trace[k, :2] * scale)], fill=age_color(k / (steps - 1)), width=3)
        x, y = trace[0, :2] * scale
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(148, 0, 211), outline=(255, 255, 255))
    return titled(canvas, title)


def compare_episode(teacher: TrackTeacher, path: Path, out: Path, seconds: float = 2.0, context: int = 10,
                    fps: float = DEFAULT_FPS) -> dict:
    episode = TraceExtractEpisode(path, fps)
    steps = int(round(seconds * fps)) + 1
    slot, keep = densest_slot(episode, steps, context)
    if slot is None or keep.sum() < 4:
        return {"episode": path.name, "skipped": True}
    frame = int(episode.frames[slot])
    lo, hi = int(episode.offsets[slot]), int(episode.offsets[slot + 1])
    theirs = np.asarray(episode.future[lo:hi], dtype=np.float64)[keep][:, :steps]
    start, stop = max(0, frame - context), min(len(episode.images), frame + int(6 * fps))
    tracks = teacher.track(np.asarray(episode.images[start:stop]), theirs[:, 0, :2],
                           np.full(len(theirs), frame - start, dtype=np.float32))
    rel = frame - start
    ours = project_tracks(tracks.points_world[:, rel : rel + steps], tracks.world_to_camera[rel], tracks.intrinsics[rel])
    error = np.linalg.norm(ours[..., :2] - theirs[..., :2], axis=-1)
    motion = lambda a: np.linalg.norm(a[:, -1, :2] - a[:, 0, :2], axis=-1)
    jitter = lambda a: float(np.median(np.linalg.norm(a[:, 2:, :2] - 2 * a[:, 1:-1, :2] + a[:, :-2, :2], axis=-1)))
    cosine = np.sum((theirs[:, -1, :2] - theirs[:, 0, :2]) * (ours[:, -1, :2] - ours[:, 0, :2]), -1)
    cosine = cosine / np.maximum(motion(theirs) * motion(ours), 1e-6)
    image = np.asarray(episode.images[frame])
    hstack([trace_panel(image, theirs, "TraceExtract"), trace_panel(image, ours, "ours")]).save(out / f"{path.name}.png")
    return {"episode": path.name, "frame": frame, "points": int(len(theirs)), "start_px": float(np.median(error[:, 0])),
            "end_px": float(np.median(error[:, -1])), "their_motion_px": float(np.median(motion(theirs))),
            "our_motion_px": float(np.median(motion(ours))), "their_jitter_px": jitter(theirs),
            "our_jitter_px": jitter(ours), "direction_cos": float(np.median(cosine)),
            "depth_ratio": float(np.median(ours[:, 0, 2] / np.maximum(theirs[:, 0, 2], 1e-3))),
            "our_reliable": float((tracks.reliability[:, rel : rel + steps] > 0.5).mean())}


def compare_traceextract(teacher: TrackTeacher, episodes: list[Path], out: Path) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    rows = [compare_episode(teacher, path, out) for path in episodes]
    (out / "compare.json").write_text(json.dumps(rows, indent=2))
    return rows

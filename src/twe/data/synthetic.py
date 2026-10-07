from pathlib import Path

import numpy as np

from twe.config import WorldConfig
from twe.contracts import COORDINATE_CONTRACT
from twe.data.manifest import Manifest
from twe.data.shards import ShardWriter
from twe.preprocess.letterbox import letterbox, source_xy_to_uv
from twe.preprocess.normalizer import normalizer_from_dataset

COLORS = {"red": (220, 40, 40), "green": (40, 200, 60), "blue": (40, 80, 220)}
DIRECTIONS = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "up": (0.0, -1.0), "down": (0.0, 1.0)}
BACKGROUNDS = {"human_nominal": (180, 160, 130), "human_corrective": (150, 150, 170),
               "robot_nominal_video": (110, 110, 110)}
SOURCE_SHAPE = (168, 224)
MOVING = 16
MASKED = 4


def synthetic_window(rng: np.random.Generator, scene: dict, cfg: WorldConfig, window: int) -> tuple[dict, dict]:
    height, width = SOURCE_SHAPE
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = scene["background"]
    image += rng.integers(0, 12, image.shape, dtype=np.uint8)
    cx, cy = scene["center"] + rng.normal(0, 3, 2)
    half = 20
    x0, x1, y0, y1 = int(cx - half), int(cx + half), int(cy - half), int(cy + half)
    image[y0:y1, x0:x1] = COLORS[scene["color"]]
    rgb, valid, transform = letterbox(image, cfg.image_size)
    grid = (np.arange(4) + 0.5) / 4
    block_xy = np.stack(np.meshgrid(x0 + grid * 2 * half, y0 + grid * 2 * half), -1).reshape(-1, 2)
    others = []
    while len(others) < cfg.num_anchors - MOVING - MASKED:
        x, y = rng.uniform(0, width), rng.uniform(0, height)
        if not (x0 - 4 <= x <= x1 + 4 and y0 - 4 <= y <= y1 + 4):
            others.append((x, y))
    xy = np.concatenate([block_xy, np.array(others), np.zeros((MASKED, 2))])
    uv = source_xy_to_uv(xy, transform, cfg.image_size).astype(np.float32)
    mask = np.arange(cfg.num_anchors) < cfg.num_anchors - MASKED
    uv[~mask] = 0.0
    offsets = np.asarray(cfg.future_offsets)
    direction = np.array([*DIRECTIONS[scene["direction"]], 0.0])
    trace = np.zeros((cfg.num_anchors, cfg.future_steps, 3), dtype=np.float32)
    trace[:MOVING] = scene["speed"] * offsets[None, :, None] * direction
    trace_valid = np.repeat(mask[:, None], cfg.future_steps, 1)
    occluded = rng.choice(np.arange(MOVING, cfg.num_anchors - MASKED), 6, replace=False)
    trace_valid[occluded, rng.integers(8, cfg.future_steps):] = False
    trace = np.where(trace_valid[..., None], trace, 0.0).astype(np.float32)
    focal = float(cfg.image_size)
    intrinsics = np.array([[focal, 0, 0], [0, focal, 0], [0, 0, 1]], dtype=np.float32)
    anchor_xyz = np.concatenate([uv, np.ones((len(uv), 1), np.float32)], 1)
    moving = np.arange(cfg.num_anchors) < MOVING
    arrays = {"anchor_xyz": anchor_xyz, "trace_moving": moving, "intrinsics": intrinsics, "rgb": rgb,
              "image_valid": valid, "anchor_uv": uv, "anchor_mask": mask, "trace": trace,
              "trace_valid": trace_valid, "trace_reliability": trace_valid.astype(np.float32)}
    meta = {"sample_id": f"synthetic/{scene['recording_id']}/{window}",
            "source": "synthetic", "recording_id": scene["recording_id"], "split_group": scene["recording_id"],
            "original_instruction": scene["instruction"],
            "instruction_available_at_t": scene["instruction"] is not None,
            "letterbox_transform": transform.to_dict(), "coordinate_contract": COORDINATE_CONTRACT,
            "geometry_provenance": "synthetic"}
    return arrays, meta


def synthetic_scene(rng: np.random.Generator, pool: str, recording_id: str, null_fraction: float) -> dict:
    color = rng.choice(list(COLORS))
    direction = rng.choice(list(DIRECTIONS))
    instruction = None if rng.random() < null_fraction else f"push the {color} block {direction}"
    center = np.array([rng.uniform(40, SOURCE_SHAPE[1] - 40), rng.uniform(40, SOURCE_SHAPE[0] - 40)])
    return {"color": color, "direction": direction, "instruction": instruction, "center": center,
            "speed": rng.uniform(0.05, 0.2), "background": BACKGROUNDS[pool], "recording_id": recording_id}


def write_synthetic(root: Path, per_shard: int = 32, windows_per_recording: int = 4, seed: int = 0) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = WorldConfig()
    rng = np.random.default_rng(seed)
    entries = []
    for split, count in (("train", per_shard), ("validation", max(8, per_shard // 4))):
        for pool in BACKGROUNDS:
            writer = ShardWriter(root, f"{split}-{pool}", pool, split)
            null_fraction = 0.3 if pool == "human_nominal" else 0.0
            for r in range(count // windows_per_recording):
                scene = synthetic_scene(rng, pool, f"{split}-{pool}-{r}", null_fraction)
                for window in range(windows_per_recording):
                    writer.add(*synthetic_window(rng, scene, cfg, window))
            entries.append(writer.close())
    Manifest(entries, "synthetic-v1").write(root)
    normalizer_from_dataset(root)
    return root

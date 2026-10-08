import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from twe.contracts import WorldContext
from twe.preprocess.letterbox import letterbox, source_xy_to_uv


@dataclass
class EpisodeGeometry:
    images: np.ndarray
    intrinsics: np.ndarray
    world_to_camera: np.ndarray
    fps: float

    def to_world(self, frame: int, pixels: np.ndarray) -> np.ndarray:
        k, z = self.intrinsics[frame], pixels[..., 2]
        camera = np.stack([(pixels[..., 0] - k[0, 2]) / k[0, 0] * z, (pixels[..., 1] - k[1, 2]) / k[1, 1] * z, z], -1)
        c2w = np.linalg.inv(self.world_to_camera[frame])
        return camera @ c2w[:3, :3].T + c2w[:3, 3]

    def to_pixels(self, frame: int, world: np.ndarray) -> np.ndarray:
        w2c, k = self.world_to_camera[frame], self.intrinsics[frame]
        camera = world @ w2c[:3, :3].T + w2c[:3, 3]
        z = np.maximum(camera[..., 2], 1e-3)
        return np.stack([k[0, 0] * camera[..., 0] / z + k[0, 2], k[1, 1] * camera[..., 1] / z + k[1, 2], z], -1)


class TraceRollout:
    def __init__(self, module, fitter, sigma, cfg, device, steps: int = 4, guidance: float = 1.0):
        self.module, self.fitter, self.cfg = module.eval(), fitter, cfg
        self.sigma = np.asarray(sigma, dtype=np.float64)
        self.device, self.steps, self.guidance = torch.device(device), steps, guidance
        self.step_seconds = cfg.horizon_seconds / cfg.future_steps
        self.seconds: list[float] = []

    def screen(self, pixels: np.ndarray, origin: np.ndarray, transform) -> np.ndarray:
        size = self.cfg.image_size
        uv = source_xy_to_uv(pixels[..., :2], transform, size) - source_xy_to_uv(origin[:, None, :2], transform, size)
        return np.concatenate([uv, np.log(pixels[..., 2:3]) - np.log(origin[:, None, 2:3])], -1)

    def pixels(self, delta: np.ndarray, origin: np.ndarray, transform) -> np.ndarray:
        size = self.cfg.image_size
        u = (source_xy_to_uv(origin[:, :2], transform, size)[:, None] + delta[..., :2]) * size
        x = (u[..., 0] - transform.offset_x) / transform.scale_x
        y = (u[..., 1] - transform.offset_y) / transform.scale_y
        return np.stack([x, y, origin[:, None, 2] * np.exp(delta[..., 2])], -1)

    @torch.no_grad()
    def predict(self, image: np.ndarray, current: np.ndarray, past: np.ndarray | None, instruction, seed: int):
        cfg, size = self.cfg, self.cfg.image_size
        rgb, valid, transform = letterbox(image, size)
        count = len(current)
        uv = torch.from_numpy(source_xy_to_uv(current[:, :2], transform, size).astype(np.float32))[None]
        history, seen = None, None
        if past is not None:
            history = torch.from_numpy((self.screen(past, current, transform) / self.sigma).astype(np.float32))[None]
            seen = torch.ones(1, count, cfg.history_steps, dtype=torch.bool)
        context = WorldContext(torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255,
                               torch.from_numpy(valid)[None], uv, torch.ones(1, count, dtype=torch.bool),
                               [instruction], history, seen).to(self.device)
        tick = time.perf_counter()
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            inputs = self.module.encode(context)
            null = self.module.encode(context.with_instructions([None])) if self.guidance != 1.0 else None
            controls = self.module.sample_controls(inputs, self.steps, torch.Generator().manual_seed(seed),
                                                   self.guidance, null)
        delta = self.fitter.decode(controls.double()).float().cpu().numpy()[0] * self.sigma
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.seconds.append(time.perf_counter() - tick)
        return self.pixels(delta, current, transform)

    def resample(self, track: np.ndarray, times: np.ndarray) -> np.ndarray:
        grid = np.arange(track.shape[1]) * self.step_seconds
        return np.stack([np.stack([np.interp(times, grid, track[n, :, d]) for d in range(3)], -1) for n in range(len(track))])

    def run(self, geometry: EpisodeGeometry, start: int, current_pixels: np.ndarray, past_pixels: np.ndarray | None,
            instruction, horizon: float, stride: float, closed_loop: bool, seed: int = 0) -> np.ndarray:
        world = geometry.to_world(start, current_pixels)
        past_world = None if past_pixels is None else geometry.to_world(start, past_pixels)
        track = [world]
        frames_per_stride = int(round(stride * geometry.fps))
        for k in range(int(round(horizon / stride))):
            frame = min(start + k * frames_per_stride, len(geometry.images) - 1) if closed_loop else start
            current = geometry.to_pixels(frame, world)
            past = None if past_world is None else geometry.to_pixels(frame, past_world)
            future = self.predict(np.asarray(geometry.images[frame]), current, past, instruction, seed + k)
            full = np.concatenate([current[:, None], future], 1)
            times = np.arange(1, frames_per_stride + 1) / geometry.fps
            segment = geometry.to_world(frame, self.resample(full, times))
            track.extend(segment.transpose(1, 0, 2))
            history_times = stride - np.arange(self.cfg.history_steps, 0, -1) * self.step_seconds
            past_world = geometry.to_world(frame, self.resample(full, np.clip(history_times, 0, None)))
            world = segment[:, -1]
        return np.stack(track, 1)

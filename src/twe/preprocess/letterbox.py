import numpy as np
import torch
import torch.nn.functional as F

from twe.contracts import LetterboxTransform


def letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray, LetterboxTransform]:
    height, width = image.shape[:2]
    scale = size / max(height, width)
    new_h = max(1, round(height * scale))
    new_w = max(1, round(width * scale))
    tensor = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].float()
    resized = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)
    resized = resized[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8).numpy()
    offset_y = (size - new_h) // 2
    offset_x = (size - new_w) // 2
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = resized
    valid = np.zeros((size, size), dtype=bool)
    valid[offset_y : offset_y + new_h, offset_x : offset_x + new_w] = True
    transform = LetterboxTransform(new_w / width, new_h / height, float(offset_x), float(offset_y), width, height)
    return canvas, valid, transform


def source_xy_to_uv(xy: np.ndarray, transform: LetterboxTransform, size: int) -> np.ndarray:
    out = np.empty_like(xy, dtype=np.float64)
    out[..., 0] = (xy[..., 0] * transform.scale_x + transform.offset_x) / size
    out[..., 1] = (xy[..., 1] * transform.scale_y + transform.offset_y) / size
    return out


def uv_to_source_xy(uv: np.ndarray, transform: LetterboxTransform, size: int) -> np.ndarray:
    out = np.empty_like(uv, dtype=np.float64)
    out[..., 0] = (uv[..., 0] * size - transform.offset_x) / transform.scale_x
    out[..., 1] = (uv[..., 1] * size - transform.offset_y) / transform.scale_y
    return out

import colorsys

import numpy as np
from PIL import Image, ImageDraw

START = (148, 0, 211)
BACKDROP = (24, 24, 28)


def age_color(frac: float) -> tuple[int, int, int]:
    hue = 0.75 * (1.0 - float(np.clip(frac, 0.0, 1.0)))
    return tuple(int(round(255 * c)) for c in colorsys.hsv_to_rgb(hue, 0.9, 0.95))


def grayscale(image: Image.Image, dim: float = 0.75) -> Image.Image:
    gray = np.asarray(image.convert("L"), dtype=np.float32) * dim
    return Image.fromarray(np.repeat(gray[..., None], 3, -1).astype(np.uint8))


def draw_paths(canvas: Image.Image, pixels: np.ndarray, valid: np.ndarray, rows: np.ndarray, upto: int,
               width: int = 3) -> Image.Image:
    draw = ImageDraw.Draw(canvas)
    steps = pixels.shape[1] - 1
    for i in rows:
        for k in range(1, min(upto, steps + 1)):
            if valid[i, k] and valid[i, k - 1]:
                draw.line([tuple(pixels[i, k - 1]), tuple(pixels[i, k])], fill=age_color(k / steps), width=width)
    for i in rows:
        x, y = pixels[i, 0]
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=START, outline=(255, 255, 255))
    return canvas


def score_color(score: float) -> tuple[int, int, int]:
    hue = 0.66 * (1.0 - float(np.clip(score, 0.0, 1.0)))
    return tuple(int(round(255 * c)) for c in colorsys.hsv_to_rgb(hue, 0.95, 1.0))


def draw_scores(canvas: Image.Image, pixels: np.ndarray, scores: np.ndarray, rows: np.ndarray) -> Image.Image:
    draw = ImageDraw.Draw(canvas)
    for i in rows:
        x, y = pixels[i]
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], fill=score_color(scores[i]), outline=(0, 0, 0))
    return canvas


def draw_history(canvas: Image.Image, pixels: np.ndarray, valid: np.ndarray, rows: np.ndarray) -> Image.Image:
    draw = ImageDraw.Draw(canvas)
    for i in rows:
        line = [tuple(p) for p, ok in zip(pixels[i], valid[i]) if ok]
        if len(line) > 1:
            draw.line(line, fill=(240, 240, 240), width=2)
    return canvas


def top_down(points: np.ndarray, valid: np.ndarray, rows: np.ndarray, upto: int, size: int,
             extent: tuple[float, float, float, float] | None = None) -> Image.Image:
    canvas = Image.new("RGB", (size, size), BACKDROP)
    draw = ImageDraw.Draw(canvas)
    selected = points[rows][valid[rows]] if len(rows) else np.zeros((0, 3))
    if extent is None:
        if len(selected) == 0:
            selected = np.array([[0.0, 0.0, 1.0]])
        lo = np.minimum(selected.min(0), [-0.2, 0, 0])
        hi = np.maximum(selected.max(0), [0.2, 0, 0.5])
        half = max(hi[0] - lo[0], hi[2] - lo[2]) / 2 * 1.15 + 1e-3
        cx, cz = (hi[0] + lo[0]) / 2, (hi[2] + lo[2]) / 2
        extent = (cx - half, cx + half, cz - half, cz + half)
    x0, x1, z0, z1 = extent
    margin = 26
    scale = (size - 2 * margin) / max(x1 - x0, z1 - z0)
    to_px = lambda p: (margin + (p[0] - x0) * scale, size - margin - (p[2] - z0) * scale)
    for frac in (0.25, 0.5, 0.75):
        y = size - margin - frac * (size - 2 * margin)
        draw.line([(margin, y), (size - margin, y)], fill=(48, 48, 56))
    draw.line([(margin, size - margin), (size - margin, size - margin)], fill=(90, 90, 100))
    draw.text((margin, 6), f"top-down x-z (depth up), {z1 - z0:.2f} depth units", fill=(200, 200, 210))
    draw.text((size - margin - 60, size - margin + 6), "x right", fill=(160, 160, 170))
    steps = points.shape[1] - 1
    for i in rows:
        for k in range(1, min(upto, steps + 1)):
            if valid[i, k] and valid[i, k - 1]:
                draw.line([to_px(points[i, k - 1]), to_px(points[i, k])], fill=age_color(k / steps), width=2)
        x, y = to_px(points[i, 0])
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=START)
    return canvas


def absolute_points(anchor_xyz: np.ndarray, trace: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.concatenate([anchor_xyz[:, None], anchor_xyz[:, None] + trace], 1)
    keep = np.concatenate([np.ones((len(valid), 1), bool), valid], 1)
    return points, keep


def titled(panel: Image.Image, title: str, height: int = 22) -> Image.Image:
    out = Image.new("RGB", (panel.width, panel.height + height), BACKDROP)
    out.paste(panel, (0, height))
    ImageDraw.Draw(out).text((6, 5), title[: panel.width // 6], fill=(235, 235, 240))
    return out


def hstack(panels: list[Image.Image], gap: int = 6) -> Image.Image:
    height = max(p.height for p in panels)
    out = Image.new("RGB", (sum(p.width for p in panels) + gap * (len(panels) - 1), height), BACKDROP)
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width + gap
    return out

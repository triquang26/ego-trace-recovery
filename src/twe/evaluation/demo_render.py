import numpy as np
from PIL import Image

from twe.evaluation.trace_style import absolute_points, draw_paths, draw_scores, grayscale, hstack, titled, top_down


def project(anchor_xyz: np.ndarray, trace: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    points = np.concatenate([anchor_xyz[:, None], anchor_xyz[:, None] + trace], 1)
    z = np.clip(points[..., 2], 1e-3, None)
    u = intrinsics[0, 0] * points[..., 0] / z + intrinsics[0, 2]
    v = intrinsics[1, 1] * points[..., 1] / z + intrinsics[1, 2]
    return np.stack([u, v], -1)


def shared_extent(sets: list[tuple[np.ndarray, np.ndarray]], rows: np.ndarray) -> tuple[float, ...]:
    chosen = [points[rows][keep[rows]] for points, keep in sets if len(rows)]
    stacked = np.concatenate(chosen) if chosen else np.array([[0.0, 0.0, 1.0]])
    lo, hi = stacked.min(0), stacked.max(0)
    half = max(hi[0] - lo[0], hi[2] - lo[2], 0.2) / 2 * 1.15
    cx, cz = (hi[0] + lo[0]) / 2, (hi[2] + lo[2]) / 2
    return cx - half, cx + half, cz - half, cz + half


def render_case(path, rgb: np.ndarray, anchor_xyz: np.ndarray, intrinsics: np.ndarray,
                panels: list[tuple], rows: np.ndarray, suptitle: str,
                upscale: int = 2) -> None:
    base = grayscale(Image.fromarray(rgb).resize((rgb.shape[1] * upscale, rgb.shape[0] * upscale)))
    sets = [absolute_points(anchor_xyz, panel[1], panel[2]) for panel in panels]
    extent = shared_extent(sets, rows)
    views, tops = [], []
    for (title, trace, valid, *scores), (points, keep) in zip(panels, sets):
        pixels = project(anchor_xyz, trace, intrinsics) * upscale
        view = draw_paths(base.copy(), pixels, keep, rows, keep.shape[1])
        if scores:
            view = draw_scores(view, pixels[:, 0], scores[0], rows)
        views.append(titled(view, title))
        tops.append(titled(top_down(points, keep, rows, keep.shape[1], base.width, extent), f"{title}: x-z"))
    row1, row2 = hstack(views), hstack(tops)
    sheet = Image.new("RGB", (row1.width, row1.height + row2.height + 30), (24, 24, 28))
    sheet.paste(titled(row1, suptitle, 30), (0, 0))
    sheet.paste(row2, (0, row1.height + 30))
    sheet.save(path)

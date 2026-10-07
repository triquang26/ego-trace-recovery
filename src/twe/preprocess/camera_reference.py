import numpy as np

AXIS_TO_OPENCV = {
    "opencv": np.eye(3),
    "opengl": np.diag([1.0, -1.0, -1.0]),
}


def to_opencv_camera(points_camera: np.ndarray, convention: str) -> np.ndarray:
    if convention not in AXIS_TO_OPENCV:
        raise ValueError(f"unverified camera convention: {convention}")
    return points_camera @ AXIS_TO_OPENCV[convention].T


def world_to_reference_camera(points_world: np.ndarray, world_to_camera: np.ndarray) -> np.ndarray:
    rotation = world_to_camera[:3, :3]
    translation = world_to_camera[:3, 3]
    return points_world @ rotation.T + translation


def robust_scene_scale(depth: np.ndarray, min_count: int = 64) -> float | None:
    values = depth[np.isfinite(depth) & (depth > 0)]
    if values.size < min_count:
        return None
    return float(np.median(values))


def moving_points(points_camera: np.ndarray, valid: np.ndarray, intrinsics: np.ndarray | None,
                  threshold_px: float) -> np.ndarray:
    if intrinsics is None or not np.isfinite(intrinsics).all():
        return np.zeros(len(points_camera), dtype=bool)
    z = np.clip(points_camera[..., 2], 1e-6, None)
    pixels = np.stack([intrinsics[0, 0] * points_camera[..., 0] / z + intrinsics[0, 2],
                       intrinsics[1, 1] * points_camera[..., 1] / z + intrinsics[1, 2]], -1)
    keep = np.concatenate([np.ones((len(valid), 1), bool), valid], 1)
    pixels = np.where(keep[..., None], np.nan_to_num(pixels), np.nan)
    low, high = np.nanmin(pixels, axis=1), np.nanmax(pixels, axis=1)
    return np.nan_to_num(np.linalg.norm(high - low, axis=-1)) >= threshold_px


def relative_displacements(
    points_camera: np.ndarray, reliability: np.ndarray, scale: float, reliability_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = points_camera[:, :1]
    future = points_camera[:, 1:]
    displacement = (future - origin) / scale
    finite = np.isfinite(future).all(-1) & np.isfinite(origin).all(-1)
    positive = (future[..., 2] > 0) & (origin[..., 2] > 0)
    future_reliability = np.where(np.isfinite(reliability[:, 1:]), reliability[:, 1:], 0.0)
    future_reliability = np.minimum(future_reliability, np.nan_to_num(reliability[:, :1], nan=0.0))
    valid = finite & positive & (future_reliability >= reliability_threshold)
    displacement = np.where(valid[..., None], displacement, 0.0).astype(np.float32)
    return displacement, valid, np.where(valid, future_reliability, 0.0).astype(np.float32)

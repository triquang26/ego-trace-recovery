import numpy as np

TARGET_SPACES = ("camera", "screen")
MIN_DEPTH_RATIO = 0.2


def project(points: np.ndarray, intrinsics: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    z = np.where(points[..., 2] > 1e-6, points[..., 2], np.nan)
    u = (intrinsics[0, 0] * points[..., 0] / z + intrinsics[0, 2]) / size
    v = (intrinsics[1, 1] * points[..., 1] / z + intrinsics[1, 2]) / size
    return np.stack([u, v], -1), z


def to_screen(anchor_xyz: np.ndarray, displacement: np.ndarray, valid: np.ndarray, intrinsics: np.ndarray,
              size: int) -> tuple[np.ndarray, np.ndarray]:
    origin_uv, origin_z = project(anchor_xyz, intrinsics, size)
    uv, z = project(anchor_xyz[:, None] + displacement, intrinsics, size)
    delta = np.concatenate([uv - origin_uv[:, None], (np.log(z) - np.log(origin_z)[:, None])[..., None]], -1)
    inside = ((uv > -0.5) & (uv < 1.5)).all(-1) & (z > MIN_DEPTH_RATIO * origin_z[:, None])
    ok = valid & np.isfinite(delta).all(-1) & np.nan_to_num(inside, nan=False).astype(bool)
    return np.where(ok[..., None], delta, 0.0).astype(np.float32), ok


def from_screen(anchor_xyz: np.ndarray, delta: np.ndarray, intrinsics: np.ndarray, size: int) -> np.ndarray:
    origin_uv, origin_z = project(anchor_xyz, intrinsics, size)
    uv = origin_uv[:, None] + delta[..., :2]
    z = origin_z[:, None] * np.exp(np.clip(delta[..., 2], -5.0, 5.0))
    x = (uv[..., 0] * size - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (uv[..., 1] * size - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.stack([x, y, z], -1)
    return np.nan_to_num(points - anchor_xyz[:, None]).astype(np.float32)


def convert(anchor_xyz: np.ndarray, displacement: np.ndarray, valid: np.ndarray, intrinsics: np.ndarray,
            size: int, space: str) -> tuple[np.ndarray, np.ndarray]:
    if space == "camera":
        return displacement.astype(np.float32), valid
    if space == "screen":
        return to_screen(anchor_xyz, displacement, valid, intrinsics, size)
    raise ValueError(f"unknown target space {space}")

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class TeacherTracks:
    points_world: np.ndarray
    reliability: np.ndarray
    world_to_camera: np.ndarray
    current_depth: np.ndarray
    convention: str
    revision: str
    intrinsics: np.ndarray | None = None


class TrackTeacher(Protocol):
    revision: str

    def track(self, frames: np.ndarray, query_xy: np.ndarray) -> TeacherTracks:
        ...

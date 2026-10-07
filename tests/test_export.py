import numpy as np
import torch

from twe.config import WorldConfig
from twe.data.shards import ShardWriter
from twe.preprocess.export_windows import ExportSettings, Recording, export_recording, window_starts
from twe.preprocess.teacher import TeacherTracks


class RigidTeacher:
    revision = "rigid-test"

    def __init__(self, timestamps, velocity, camera_velocity, convention):
        self.timestamps = timestamps
        self.velocity = np.asarray(velocity)
        self.camera_velocity = np.asarray(camera_velocity)
        self.convention = convention

    def track(self, frames, query_xy):
        count = len(frames)
        t = self.timestamps[:count] - self.timestamps[0]
        start = np.stack([query_xy[:, 0] / 100, query_xy[:, 1] / 100, np.full(len(query_xy), 4.0)], -1)
        points = start[:, None] + t[None, :, None] * self.velocity
        poses = np.repeat(np.eye(4)[None], count, 0)
        poses[:, :3, 3] = -t[:, None] * self.camera_velocity
        if self.convention == "opengl":
            flip = np.diag([1.0, -1.0, -1.0])
            points = points @ flip.T
            poses[:, :3, :3] = flip @ poses[:, :3, :3] @ flip.T
            poses[:, :3, 3] = poses[:, :3, 3] @ flip.T
        depth = np.full((48, 64), 4.0)
        return TeacherTracks(points, np.ones((len(query_xy), count)), poses, depth, self.convention, self.revision)


def grid_selector(rgb, valid):
    uv = torch.rand(1, 64, 2, generator=torch.Generator().manual_seed(0)) * 0.5 + 0.25
    return uv, torch.ones(1, 64, dtype=torch.bool)


def run_export(tmp_path, convention, timestamps):
    cfg = WorldConfig()
    frames = np.zeros((len(timestamps), 48, 64, 3), dtype=np.uint8)
    recording = Recording("rec", "unit", "rec", "push the cup", timestamps, lambda idx: frames[idx])
    teacher = RigidTeacher(timestamps, [0.4, 0.0, 0.0], [0.2, 0.1, 0.0], convention)
    writer = ShardWriter(tmp_path, "s", "human_nominal", "train")
    starts = window_starts(timestamps, cfg.horizon_seconds, 0.5)
    written = export_recording(recording, starts, teacher, grid_selector, cfg, ExportSettings(), writer)
    return writer, written, cfg


def test_export_matches_rigid_motion_in_both_conventions(tmp_path):
    timestamps = np.arange(0, 3.0, 1 / 30)
    for convention in ("opencv", "opengl"):
        writer, written, cfg = run_export(tmp_path / convention, convention, timestamps)
        assert written == len(writer) == 2
        trace = writer.arrays["trace"][0]
        valid = writer.arrays["trace_valid"][0]
        expected_x = 0.4 * np.asarray(cfg.future_offsets) / 4.0
        assert valid.all()
        assert np.allclose(trace[:, :, 0], expected_x[None], atol=1e-5)
        assert np.allclose(trace[:, :, 1:], 0, atol=1e-6)


def test_export_marks_gap_invalid(tmp_path):
    timestamps = np.concatenate([np.arange(0, 1.0, 1 / 30), np.arange(1.6, 3.0, 1 / 30)])
    writer, written, cfg = run_export(tmp_path, "opencv", timestamps)
    valid = writer.arrays["trace_valid"][0][0]
    offsets = np.asarray(cfg.future_offsets)
    assert not valid[(offsets > 1.0) & (offsets < 1.6)].any()
    assert valid[offsets < 0.95].all()

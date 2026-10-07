import numpy as np
import torch

from twe.config import WorldConfig
from twe.data.shards import ShardWriter
from twe.preprocess.chunking import plan_chunks, window_starts
from twe.preprocess.export_windows import ExportSettings, Recording, export_recording
from twe.preprocess.teacher import TeacherTracks


class RigidTeacher:
    revision = "rigid-test"

    def __init__(self, timestamps, velocity, camera_velocity, convention):
        self.timestamps = timestamps
        self.velocity = np.asarray(velocity)
        self.camera_velocity = np.asarray(camera_velocity)
        self.convention = convention

    def track(self, frames, query_xy, query_frame):
        count = len(frames)
        t = self.timestamps[:count] - self.timestamps[0]
        tq = t[query_frame.astype(int)]
        start = np.stack([query_xy[:, 0] / 100, query_xy[:, 1] / 100, np.full(len(query_xy), 4.0)], -1)
        start = start + tq[:, None] * self.camera_velocity
        points = start[:, None] + (t[None, :] - tq[:, None])[..., None] * self.velocity
        poses = np.repeat(np.eye(4)[None], count, 0)
        poses[:, :3, 3] = -t[:, None] * self.camera_velocity
        if self.convention == "opengl":
            flip = np.diag([1.0, -1.0, -1.0])
            points = points @ flip.T
            poses[:, :3, :3] = flip @ poses[:, :3, :3] @ flip.T
            poses[:, :3, 3] = poses[:, :3, 3] @ flip.T
        depth = {int(f): np.full((48, 64), 4.0) for f in np.unique(query_frame)}
        return TeacherTracks(points, np.ones((len(query_xy), count)), poses, depth, self.convention, self.revision)


def grid_selector(rgb, valid):
    uv = torch.rand(1, 64, 2, generator=torch.Generator().manual_seed(0)) * 0.5 + 0.25
    return uv.expand(len(rgb), -1, -1), torch.ones(len(rgb), 64, dtype=torch.bool)


def run_export(tmp_path, convention, timestamps, chunk_seconds=12.0):
    cfg = WorldConfig()
    frames = np.zeros((len(timestamps), 48, 64, 3), dtype=np.uint8)
    recording = Recording("rec", "unit", "rec", "push the cup", timestamps, lambda idx: frames[idx])
    teacher = RigidTeacher(timestamps, [0.4, 0.0, 0.0], [0.2, 0.1, 0.0], convention)
    writer = ShardWriter(tmp_path, "s", "human_nominal", "train")
    settings = ExportSettings(chunk_seconds=chunk_seconds)
    written = export_recording(recording, teacher, grid_selector, cfg, settings, writer, 0.5)
    return writer, written, cfg


def test_export_matches_rigid_motion_in_both_conventions(tmp_path):
    timestamps = np.arange(0, 3.0, 1 / 30)
    for convention, chunk in (("opencv", 12.0), ("opengl", 12.0), ("opencv", 2.2)):
        writer, written, cfg = run_export(tmp_path / f"{convention}{chunk}", convention, timestamps, chunk)
        assert written == len(writer) == 2
        expected_x = 0.4 * np.asarray(cfg.future_offsets) / 4.0
        for row in range(2):
            trace = writer.arrays["trace"][row]
            assert writer.arrays["trace_valid"][row].all()
            assert np.allclose(trace[:, :, 0], expected_x[None], atol=1e-5)
            assert np.allclose(trace[:, :, 1:], 0, atol=1e-6)


def test_plan_chunks_keeps_every_window_inside_one_chunk():
    timestamps = np.arange(0, 30.0, 1 / 15)
    starts = window_starts(timestamps, 2.1, 1.0)
    chunks = plan_chunks(timestamps, starts, 2.1, 8.0)
    covered = [chunk.frames[s] for chunk in chunks for s in chunk.starts]
    assert covered == starts
    for chunk in chunks:
        span = timestamps[chunk.frames[-1]] - timestamps[chunk.frames[0]]
        assert span <= 8.0 + 1e-9
        assert all(timestamps[chunk.frames[-1]] >= timestamps[chunk.frames[s]] + 2.1 - 1 / 15 for s in chunk.starts)


def test_export_marks_gap_invalid(tmp_path):
    timestamps = np.concatenate([np.arange(0, 1.0, 1 / 30), np.arange(1.6, 3.0, 1 / 30)])
    writer, written, cfg = run_export(tmp_path, "opencv", timestamps)
    valid = writer.arrays["trace_valid"][0][0]
    offsets = np.asarray(cfg.future_offsets)
    assert not valid[(offsets > 1.0) & (offsets < 1.6)].any()
    assert valid[offsets < 0.95].all()


class LimitedTeacher(RigidTeacher):
    def __init__(self, *args, max_queries):
        super().__init__(*args)
        self.max_queries = max_queries
        self.calls = 0

    def track(self, frames, query_xy, query_frame):
        self.calls += 1
        if len(query_xy) > self.max_queries:
            raise torch.OutOfMemoryError("too many queries")
        return super().track(frames, query_xy, query_frame)


def test_export_splits_chunk_on_out_of_memory(tmp_path):
    timestamps = np.arange(0, 6.0, 1 / 30)
    cfg = WorldConfig()
    frames = np.zeros((len(timestamps), 48, 64, 3), dtype=np.uint8)
    recording = Recording("rec", "unit", "rec", "push the cup", timestamps, lambda idx: frames[idx])
    teacher = LimitedTeacher(timestamps, [0.4, 0.0, 0.0], [0.0, 0.0, 0.0], "opencv", max_queries=128)
    writer = ShardWriter(tmp_path, "s", "human_nominal", "train")
    written = export_recording(recording, teacher, grid_selector, cfg, ExportSettings(), writer, 0.5)
    assert written == 8 and teacher.calls > 1
    expected_x = 0.4 * np.asarray(cfg.future_offsets) / 4.0
    assert np.allclose(writer.arrays["trace"][:, :, :, 0], expected_x, atol=1e-5)

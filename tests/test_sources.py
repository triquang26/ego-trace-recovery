import av
import h5py
import numpy as np
import torch

from twe.data.egodex import egodex_recordings
from twe.data.manifest import Manifest
from twe.data.shards import ShardWriter
from twe.data.video import VideoFile
from twe.preprocess.spatracker_teacher import SpaTrackerTeacher


def write_video(path, count, fps=30):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for index in range(count):
            image = np.full((48, 64, 3), index * 8 % 256, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_video_reader_returns_requested_frames(tmp_path):
    write_video(tmp_path / "v.mp4", 20)
    video = VideoFile(tmp_path / "v.mp4")
    assert len(video) == 20 and np.allclose(np.diff(video.timestamps), 1 / 30)
    frames = video.read(np.array([5, 6, 12]))
    assert frames.shape == (3, 48, 64, 3)
    assert abs(int(frames[2].mean()) - 96) <= 4


def test_egodex_recordings_use_selected_description(tmp_path):
    task = tmp_path / "test" / "pour_water"
    task.mkdir(parents=True)
    write_video(task / "0.mp4", 10)
    with h5py.File(task / "0.hdf5", "w") as handle:
        handle.attrs["llm_description"] = "pour water into the cup"
        handle.attrs["llm_description2"] = "pour water back into the bottle"
        handle.attrs["which_llm_description"] = 2
    (recording,) = list(egodex_recordings(tmp_path / "test"))
    assert recording.instruction == "pour water back into the bottle"
    assert recording.recording_id == "test/pour_water/0" and len(recording.timestamps) == 10
    (half,) = list(egodex_recordings(tmp_path / "test", frame_step=2))
    assert np.allclose(half.timestamps, recording.timestamps[::2])
    assert np.array_equal(half.read_frames(np.array([1, 2])), recording.read_frames(np.array([2, 4])))


def test_manifest_collect_reads_shard_entries(tmp_path):
    arrays = {"anchor_xyz": np.zeros((2, 3)), "intrinsics": np.eye(3), "trace_moving": np.zeros(2, bool),
              "rgb": np.zeros((4, 4, 3)), "image_valid": np.ones((4, 4), bool), "anchor_uv": np.zeros((2, 2)),
              "anchor_mask": np.ones(2, bool), "trace": np.zeros((2, 3, 3)), "trace_valid": np.ones((2, 3), bool),
              "trace_reliability": np.ones((2, 3))}
    for name, split in (("a", "train"), ("b", "validation")):
        writer = ShardWriter(tmp_path, name, "human_nominal", split)
        writer.add(arrays, {"recording_id": name})
        writer.close()
    manifest = Manifest.collect(tmp_path, "teacher-x")
    assert [e.path for e in manifest.select("train")] == ["a"]
    assert Manifest.read(tmp_path).teacher_revision == "teacher-x"


class StubFront(torch.nn.Module):
    def forward(self, video):
        t, h, w = video.shape[1], video.shape[-2], video.shape[-1]
        return {"poses_pred": torch.eye(4).repeat(1, t, 1, 1), "intrs": torch.eye(3).repeat(1, t, 1, 1),
                "points_map": torch.ones(1, t, h, w, 3), "unc_metric": torch.ones(1, t, h, w)}


class StubPredictor:
    def __init__(self, sign):
        self.sign = sign

    def forward(self, video, queries, **kwargs):
        t, n = len(video), len(queries)
        h, w = video.shape[-2:]
        focal = 100.0
        intrs = torch.tensor([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.0]]).repeat(t, 1, 1)
        z = torch.full((n,), 2.0)
        x = (torch.from_numpy(queries[:, 1]) - w / 2) * z / focal
        y = (torch.from_numpy(queries[:, 2]) - h / 2) * z / focal
        camera = torch.stack([x, y, self.sign * z], -1)
        track3d = torch.cat([camera, torch.zeros(n, 3)], -1).repeat(t, 1, 1)
        c2w = torch.eye(4).repeat(t, 1, 1)
        c2w[:, 0, 3] = torch.arange(t) * 0.1
        point_map = torch.full((t, 3, h, w), 2.0)
        ones = torch.ones(t, n, 1)
        return c2w, intrs, point_map, torch.ones(t, h, w), track3d, None, ones, ones * 0.9, video


def resize(video):
    return torch.nn.functional.interpolate(video, size=(28, 56))


def test_spatracker_teacher_verifies_convention_and_outputs_world_tracks():
    frames = np.zeros((5, 48, 96, 3), dtype=np.uint8)
    query = np.array([[10.0, 20.0], [50.0, 30.0]])
    when = np.array([0, 2])
    teacher = SpaTrackerTeacher(StubFront(), StubPredictor(1.0), resize, width=56, device="cpu")
    tracks = teacher.track(frames, query, when)
    assert sorted(tracks.depth) == [0, 2] and tracks.intrinsics.shape == (5, 3, 3)
    assert tracks.points_world.shape == (2, 5, 3) and tracks.convention == "opencv"
    assert np.allclose(tracks.points_world[:, 4, 0] - tracks.points_world[:, 0, 0], 0.4)
    assert np.allclose(tracks.reliability, 0.9) and np.allclose(tracks.world_to_camera[1, 0, 3], -0.1)
    flipped = SpaTrackerTeacher(StubFront(), StubPredictor(-1.0), resize, width=56, device="cpu")
    try:
        flipped.track(frames, query, when)
        raise AssertionError("expected convention failure")
    except ValueError as error:
        assert "reprojection" in str(error)

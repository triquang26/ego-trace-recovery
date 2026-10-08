import json

import numpy as np
import torch

from synthetic import TINY_WORLD, tiny_module, tiny_stage1
from test_benchmark import fake_episode
from test_model import make_context
from twe.data.dataset import QuerySampling, WorldWindowDataset
from twe.data.synthetic import write_synthetic
from twe.evaluation.demo import DemoBuilder
from twe.evaluation.prompt_sweep import build_prompt_sweep
from twe.evaluation.teacher_compare import compare_traceextract
from twe.preprocess.teacher import TeacherTracks
from twe.training.pretrain_world import make_fitter


class PinholeTeacher:
    revision = "pinhole-test"

    def __init__(self, speed: float):
        self.speed = speed
        self.k = np.array([[200.0, 0, 160.0], [0, 200.0, 90.0], [0, 0, 1.0]])

    def track(self, frames, query_xy, query_frame):
        count = len(frames)
        t = np.arange(count)[None, :] - query_frame[:, None]
        x = query_xy[:, :1] + self.speed * t
        y = np.repeat(query_xy[:, 1:2], count, 1)
        points = np.stack([(x - 160.0) / 200.0, (y - 90.0) / 200.0, np.ones_like(x)], -1)
        poses = np.repeat(np.eye(4)[None], count, 0)
        return TeacherTracks(points, np.ones((len(query_xy), count)), poses, np.ones((count, 4, 4)), "opencv",
                             self.revision, np.repeat(self.k[None], count, 0))


def test_guidance_one_matches_plain_sampling_and_amplifies_text():
    module = tiny_module().eval()
    context = make_context()
    text = module.encode(context)
    null = module.encode(context.with_instructions([None, None]))
    plain = module.sample_controls(text, 2, torch.Generator().manual_seed(0))
    same = module.sample_controls(text, 2, torch.Generator().manual_seed(0), 1.0, null)
    strong = module.sample_controls(text, 2, torch.Generator().manual_seed(0), 4.0, null)
    assert torch.allclose(plain, same) and not torch.allclose(plain[0], strong[0])
    probs = module.motion_probability(text, 3.0, null)
    assert probs.shape == (2, 64) and ((probs >= 0) & (probs <= 1)).all()


def test_prompt_sweep_writes_panels(tmp_path):
    root = write_synthetic(tmp_path / "data")
    fitter = make_fitter(tiny_stage1())
    dataset = WorldWindowDataset(root, "validation", [1, 1, 1], fitter, QuerySampling(64, 4, False))
    builder = DemoBuilder(tiny_module(), fitter, dataset, [1, 1, 1], "cpu", steps=2)
    record = build_prompt_sweep(builder, tmp_path / "prompts", scenes=2, guidances=(1.0, 3.0))
    assert len(record["cases"]) == 4 and all((tmp_path / "prompts" / c["image"]).exists() for c in record["cases"])
    assert all(len(c["shift_vs_original"]) == 4 for c in record["cases"])
    assert json.loads((tmp_path / "prompts" / "prompts.json").read_text())[0]["guidance"] == 1.0


def test_teacher_compare_agrees_with_matching_tracker(tmp_path):
    path = fake_episode(tmp_path, name="test_dataset_egodex/ep0", frames=50, speed=3.0, frame=12)
    rows = compare_traceextract(PinholeTeacher(3.0), [path], tmp_path / "out")
    row = rows[0]
    assert row["points"] == 6 and row["end_px"] < 0.5 and row["direction_cos"] > 0.99
    assert abs(row["their_motion_px"] - row["our_motion_px"]) < 0.5 and (tmp_path / "out" / "ep0.png").exists()
    slow = compare_traceextract(PinholeTeacher(1.0), [path], tmp_path / "slow")[0]
    assert slow["our_motion_px"] < slow["their_motion_px"] and slow["end_px"] > 10

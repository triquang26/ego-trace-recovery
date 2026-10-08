import json
from types import SimpleNamespace

import numpy as np
import pytest

from synthetic import TINY_WORLD, tiny_module, tiny_stage1
from twe.data.dataset import QuerySampling, TaskSelection, WorldWindowDataset
from twe.data.synthetic import write_synthetic
from twe.data.traceextract import TraceExtractEpisode, traceextract_groups, traceextract_items
from twe.evaluation.benchmark import METRICS, BenchmarkRunner, egodex_groups
from twe.training.pretrain_world import make_fitter


def fake_episode(root, name="test_dataset_droid/ep0", frames=40, points=6, height=180, width=320, speed=4.0):
    path = root / name
    (path / "samples").mkdir(parents=True)
    np.save(path / "images.npy", np.full((frames, height, width, 3), 90, dtype=np.uint8))
    starts = np.stack([np.linspace(60, 260, points), np.full(points, 90.0), np.full(points, 1.0)], -1)
    steps = np.arange(frames)[None, :, None] * np.array([speed, 0.0, 0.0])
    future = (starts[:, None] + steps).astype(np.float16)
    history = (starts[:, None] - steps).astype(np.float16)
    np.save(path / "samples/frame_indices.npy", np.array([0], dtype=np.int32))
    np.save(path / "samples/offsets.npy", np.array([0, points], dtype=np.int64))
    np.save(path / "samples/raw_traj.npy", future)
    np.save(path / "samples/raw_traj_history.npy", history)
    np.save(path / "samples/raw_valid_steps.npy", np.ones((points, frames), dtype=bool))
    np.save(path / "samples/raw_valid_steps_history.npy", np.ones((points, frames), dtype=bool))
    (path / "description.txt").write_text("push the blocks right\n")
    return path


def test_task_selection_filters_windows_and_shards(tmp_path):
    root = write_synthetic(tmp_path / "data")
    fitter = make_fitter(tiny_stage1())
    full = WorldWindowDataset(root, "train", [1, 1, 1], fitter, QuerySampling())
    none = WorldWindowDataset(root, "train", [1, 1, 1], fitter, QuerySampling(), "screen",
                              TaskSelection(only=frozenset({"missing"})))
    assert len(full) > 0 and len(none) == 0
    entries = [SimpleNamespace(path=f"shard-{i}") for i in range(9)]
    half = TaskSelection(shard_fraction=0.5)
    assert len(half.shards(entries)) == 5 and half.shards(entries) == half.shards(list(reversed(entries)))[::-1]
    assert TaskSelection(exclude=frozenset({"a"})).allows("b") and not TaskSelection(exclude=frozenset({"a"})).allows("a")


def test_traceextract_item_matches_screen_motion(tmp_path):
    fake_episode(tmp_path)
    assert list(traceextract_groups(tmp_path)) == ["droid"]
    episode = TraceExtractEpisode(tmp_path / "test_dataset_droid/ep0", fps=5.0)
    item = episode.item(0, TINY_WORLD, make_fitter(tiny_stage1()), [1.0, 1.0, 1.0], np.random.default_rng(0))
    count = int(item["anchor_mask"].sum())
    assert count == 6 and item["instruction"] == "push the blocks right"
    scale = TINY_WORLD.image_size / 320
    expected = 4.0 * 5.0 * 2.0 * scale / TINY_WORLD.image_size
    assert item["trace_valid"][:count].all()
    assert np.allclose(item["trace"][:count, -1, 0].numpy(), expected, atol=1e-3)
    assert np.allclose(item["trace"][:count, :, 1:].numpy(), 0.0, atol=1e-3)
    assert (item["history"][:count, 0, 0] < 0).all() and item["history_valid"][:count].all()


def test_benchmark_reports_every_group(tmp_path):
    fake_episode(tmp_path / "mu0")
    root = write_synthetic(tmp_path / "data")
    fitter = make_fitter(tiny_stage1())
    groups = egodex_groups(root, TINY_WORLD, fitter, [1, 1, 1], frozenset(), limit=8)
    groups.update(traceextract_items(tmp_path / "mu0", TINY_WORLD, fitter, [1, 1, 1], per_episode=1))
    report = BenchmarkRunner(tiny_module(), fitter, [1, 1, 1], "cpu", samples=2, steps=2, batch=4).run(groups)
    assert set(report) == {"egodex_val_seen_tasks", "mu0_droid"}
    assert all(set(METRICS) <= set(r) for r in report.values())
    assert report["mu0_droid"]["motion_auroc"] is None and report["egodex_val_seen_tasks"]["windows"] == 8
    json.dumps(report)
    assert report["mu0_droid"]["zero"] == pytest.approx(4.0 * 5.0 * 100 / 320 * np.mean(np.arange(1, 33) / 32 * 2), rel=1e-2)

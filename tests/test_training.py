import json

import torch

from synthetic import tiny_module, tiny_stage1
from twe.data.synthetic import write_synthetic
from twe.data.dataset import WorldWindowDataset
from twe.data.sampler import MixtureBatchSampler
from twe.training.checkpoint import load_world
from twe.training.pretrain_world import make_fitter, train


def test_sampler_mixture_and_null_cap(tmp_path):
    root = write_synthetic(tmp_path / "data")
    cfg = tiny_stage1()
    dataset = WorldWindowDataset(root, "train", [1, 1, 1], make_fitter(cfg))
    metas = [dataset.meta(i) for i in range(len(dataset))]
    sampler = MixtureBatchSampler(metas, cfg.mixture, 20, 0.2, 200, 0)
    pools, nulls = {}, []
    for batch in sampler:
        nulls.append(sum(not metas[i]["original_instruction"] for i in batch))
        for i in batch:
            pools[metas[i]["pool"]] = pools.get(metas[i]["pool"], 0) + 1
    assert max(nulls) <= 4
    total = sum(pools.values())
    assert abs(pools["human_nominal"] / total - 0.5) < 0.03


def test_dataset_targets_align_with_anchors(tmp_path):
    root = write_synthetic(tmp_path / "data")
    cfg = tiny_stage1()
    dataset = WorldWindowDataset(root, "train", [1, 1, 1], make_fitter(cfg))
    item = dataset[1]
    assert item["controls"].shape == (64, 10, 3) and item["fit_valid"][:16].all()
    moving = item["trace"].abs().sum((-1, -2)) > 0
    assert moving[:16].all() and not moving[16:].any()


def test_training_reduces_loss_and_checkpoint_loads(tmp_path):
    root = write_synthetic(tmp_path / "data")
    out = tmp_path / "run"
    cfg = tiny_stage1()
    calls = []
    metrics = train(cfg, root, out, module=tiny_module(), device="cpu", on_checkpoint=calls.append)
    records = [json.loads(line) for line in open(out / "log.jsonl")]
    losses = [r["loss"] for r in records if "loss" in r]
    assert losses[-1] < 0.7 * losses[0]
    assert metrics["ade"] < metrics["zero_motion_ade"] + 1.0 and len(calls) == 2
    report = json.loads((out / "parameters.json").read_text())
    assert report["trainable"] == report["trace_expert"]
    module = tiny_module()
    artifact = load_world(out / "world_latest.pt", module)
    assert module.model_revision == artifact["model_revision"]
    resumed = tiny_stage1(optimizer_updates=70)
    train(resumed, root, out, module=tiny_module(), device="cpu")
    state = torch.load(out / "train_state.pt", weights_only=False)
    assert state["update"] == 70

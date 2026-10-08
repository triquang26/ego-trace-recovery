import json
from dataclasses import replace

import pytest
import torch

from synthetic import TINY_WORLD, grounding_layers, tiny_module, tiny_stage1
from test_model import make_context
from twe.data.synthetic import write_synthetic
from twe.models.fusion import PromptFeatures, fusion_layers_from
from twe.preprocess.current_anchors import pool_visual
from twe.training.checkpoint import load_world
from twe.training.ema import ExponentialAverage
from twe.training.objective import masked_flow_loss, motion_loss
from twe.training.pretrain_world import train

FUSED_WORLD = replace(TINY_WORLD, fusion_layers=2, pooled_grid=12)


def prompt(batch=2, length=6, padded=2):
    torch.manual_seed(4)
    mask = torch.ones(batch, length, dtype=torch.bool)
    mask[1, length - padded:] = False
    self_mask = (mask[:, :, None] & mask[:, None, :]) | torch.eye(length, dtype=torch.bool)
    return PromptFeatures(torch.randn(batch, length, 256), mask, self_mask, torch.zeros(batch, length, 256))


def test_fusion_layers_copy_grounding_weights_and_are_trainable():
    source = grounding_layers(3)
    copies = fusion_layers_from(source, 2)
    assert len(copies) == 2
    for layer, original in zip(copies, source):
        for mine, theirs in ((layer.fusion, original.fusion_layer), (layer.enhancer, original.text_enhancer_layer)):
            for (name, a), (_, b) in zip(mine.named_parameters(), theirs.named_parameters()):
                assert torch.equal(a, b) and a.requires_grad and a.data_ptr() != b.data_ptr(), name
    with pytest.raises(ValueError):
        fusion_layers_from(source, 4)


def test_fusion_ignores_padding_on_both_streams():
    layer = fusion_layers_from(grounding_layers(1), 1)[0].eval()
    text = prompt()
    visual = torch.randn(2, 10, 256)
    pad = torch.zeros(2, 10, dtype=torch.bool)
    pad[:, 7:] = True
    v, t = layer(visual, pad, text.features, text)
    noisy_visual = visual.clone()
    noisy_visual[:, 7:] = 50.0
    noisy_text = text.features.clone()
    noisy_text[1, 4:] = 50.0
    v2, t2 = layer(noisy_visual, pad, noisy_text, PromptFeatures(noisy_text, text.mask, text.self_mask,
                                                                 text.positions))
    assert torch.allclose(v[:, :7], v2[:, :7], atol=1e-4) and torch.allclose(t[1, :4], t2[1, :4], atol=1e-4)


def test_fused_world_conditions_points_on_instruction_and_trains_fusion():
    module = tiny_module(FUSED_WORLD).eval()
    context = make_context()
    inputs = module.encode(context)
    assert inputs.visual.shape == (2, 144, 256) and inputs.fused_local.shape == (2, 64, 256)
    noisy, s = torch.randn(2, 64, 10, 3), torch.full((2,), 0.5)
    base = module(inputs, noisy, s).velocity
    swapped = context.with_instructions(["pick up the red cup", "open the drawer"])
    assert not torch.allclose(base, module(module.encode(swapped), noisy, s).velocity, atol=1e-4)
    module.train()
    out = module(module.encode(context), noisy, s)
    loss = masked_flow_loss(out.velocity, torch.randn_like(noisy), context.anchor_mask)
    (loss + motion_loss(out.motion_logits, torch.ones(2, 64, dtype=torch.bool), context.anchor_mask)).backward()
    assert all(p.grad is None for p in module.visual_encoder.parameters())
    assert all(p.grad is None for p in module.grounding_encoder.parameters())
    assert all(p.grad is not None for p in module.fusion.parameters())
    with pytest.raises(ValueError):
        type(module)(FUSED_WORLD, module.visual_encoder, module.grounding_encoder)


def test_pool_visual_handles_non_divisible_grid():
    features = torch.randn(2, 21, 21, 8)
    weight = torch.ones(2, 21, 21)
    weight[1, :5] = 0
    tokens, mask = pool_visual(features, weight, 12)
    assert tokens.shape == (2, 144, 8) and mask[0].all() and not mask[1, 0] and torch.isfinite(tokens).all()
    exact, _ = pool_visual(features[:, :20, :20], torch.ones(2, 20, 20), 10)
    assert torch.allclose(exact[0, 0], features[0, :2, :2].mean((0, 1)), atol=1e-6)


def test_ema_tracks_and_restores_parameters():
    param = torch.nn.Parameter(torch.zeros(3))
    ema = ExponentialAverage([param], 0.5)
    for value in (2.0, 4.0):
        param.data.fill_(value)
        ema.update()
    averaged = ema.shadow[0].clone()
    assert (averaged > 0).all() and (averaged < 4.0).all()
    with ema.applied():
        assert torch.equal(param.data, averaged)
    assert (param.data == 4.0).all()
    clone = ExponentialAverage([torch.nn.Parameter(torch.zeros(3))], 0.5)
    clone.load_state_dict(ema.state_dict())
    assert torch.equal(clone.shadow[0], averaged) and clone.updates == 2


def test_fused_training_with_ema_checkpoints_and_resumes(tmp_path):
    root = write_synthetic(tmp_path / "data")
    out = tmp_path / "run"
    cfg = replace(tiny_stage1(optimizer_updates=30, eval_every=15, checkpoint_every=15), world=FUSED_WORLD,
                  ema_decay=0.9)
    metrics = train(cfg, root, out, module=tiny_module(FUSED_WORLD), device="cpu")
    assert {"flow_null_text", "flow_shuffled_text"} <= set(metrics)
    report = json.loads((out / "parameters.json").read_text())
    assert report["trainable"] == report["trace_expert"] + report["fusion"] > report["trace_expert"]
    module = tiny_module(FUSED_WORLD)
    artifact = load_world(out / "world_latest.pt", module)
    assert set(artifact["trainable_state"]) == {"expert", "fusion"}
    train(replace(cfg, optimizer_updates=35), root, out, module=tiny_module(FUSED_WORLD), device="cpu")
    state = torch.load(out / "train_state.pt", weights_only=False)
    assert state["update"] == 35 and state["ema"]["updates"] == 35

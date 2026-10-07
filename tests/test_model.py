import torch

from synthetic import TINY_WORLD, tiny_module
from twe.contracts import WorldContext
from twe.preprocess.current_anchors import select_anchors
from twe.training.objective import masked_flow_loss, validity_loss


def make_context(batch=2, points=64):
    torch.manual_seed(3)
    rgb = torch.rand(batch, 3, 224, 224)
    valid = torch.ones(batch, 224, 224, dtype=torch.bool)
    valid[1, :40] = False
    valid[1, -40:] = False
    rgb[~valid[:, None].expand_as(rgb)] = 0
    uv = torch.rand(batch, points, 2) * 0.6 + 0.2
    mask = torch.ones(batch, points, dtype=torch.bool)
    return WorldContext(rgb, valid, uv, mask, ["place the bowl on the plate", None])


def test_shapes_and_permutation_equivariance():
    module = tiny_module().eval()
    context = make_context()
    inputs = module.encode(context)
    assert inputs.visual.shape == (2, 64, 384) and inputs.anchor_features.shape == (2, 64, 384)
    noisy = torch.randn(2, 64, 10, 3)
    s = torch.tensor([0.3, 1.0])
    out = module(inputs, noisy, s)
    assert out.hidden.shape == (2, 64, TINY_WORLD.width)
    assert out.velocity.shape == (2, 64, 10, 3) and out.validity_logits.shape == (2, 64, 32)
    perm = torch.randperm(64)
    context_p = WorldContext(context.rgb, context.image_valid, context.anchor_uv[:, perm],
                             context.anchor_mask[:, perm], context.instructions)
    out_p = module(module.encode(context_p), noisy[:, perm], s)
    assert torch.allclose(out_p.velocity, out.velocity[:, perm], atol=1e-5)


def test_padded_anchors_do_not_change_valid_outputs():
    module = tiny_module().eval()
    context = make_context()
    noisy = torch.randn(2, 64, 10, 3)
    s = torch.full((2,), 0.5)
    context.anchor_mask[:, 48:] = False
    base = module(module.encode(context), noisy, s).velocity
    noisy2 = noisy.clone()
    noisy2[:, 48:] = 1e3
    context.anchor_uv[:, 48:] = 0.9
    changed = module(module.encode(context), noisy2, s).velocity
    assert torch.allclose(base[:, :48], changed[:, :48], atol=1e-5)


def test_only_expert_receives_gradients():
    module = tiny_module()
    context = make_context()
    inputs = module.encode(context)
    out = module(inputs, torch.randn(2, 64, 10, 3), torch.rand(2))
    loss = masked_flow_loss(out.velocity, torch.randn(2, 64, 10, 3), context.anchor_mask)
    loss = loss + validity_loss(out.validity_logits, torch.ones(2, 64, 32), context.anchor_mask)
    loss.backward()
    assert all(p.grad is None for p in module.visual_encoder.parameters())
    assert all(p.grad is None for p in module.text_encoder.parameters())
    grads = [p.grad for p in module.expert.parameters() if p.requires_grad]
    assert sum(g is not None for g in grads) == len(grads)


def test_anchor_selection_current_only_and_deterministic():
    module = tiny_module()
    context = make_context()
    uv, mask = module.select_anchors(context.rgb, context.image_valid)
    uv2, _ = module.select_anchors(context.rgb.clone(), context.image_valid)
    assert torch.equal(uv, uv2) and uv.shape == (2, 64, 2)
    rows = (uv[1, mask[1], 1] * 224).long()
    assert context.image_valid[1, rows, 112].all()
    extra = uv[0, 32:][mask[0, 32:]]
    assert torch.cdist(extra, extra).add(torch.eye(len(extra)) * 9).min() >= TINY_WORLD.anchor_min_distance - 1e-6
    patches = module.visual_encoder(context.rgb)
    direct = select_anchors(patches, context.image_valid, 64, 32, TINY_WORLD.anchor_min_distance)
    assert torch.equal(direct[0], uv)


def test_extract_features_contract():
    module = tiny_module().eval()
    context = make_context()
    a = module.extract_features(context.rgb, context.image_valid, context.instructions)
    b = module.extract_features(context.rgb, context.image_valid, context.instructions)
    assert a.hidden.shape == (2, 64, TINY_WORLD.width) and torch.equal(a.hidden, b.hidden)
    assert a.anchor_mask.dtype == torch.bool and a.noise_protocol

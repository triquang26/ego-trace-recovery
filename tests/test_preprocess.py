import numpy as np
import torch

from twe.preprocess.bspline_targets import BSplineTargets
from twe.preprocess.camera_reference import relative_displacements, to_opencv_camera, world_to_reference_camera
from twe.preprocess.letterbox import letterbox, source_xy_to_uv, uv_to_source_xy
from twe.preprocess.temporal_sampling import gather_tracks, plan_future_samples


def make_fitter(**kw):
    args = dict(future_steps=32, free_controls=10, degree=3, regularization=1e-3, min_valid_steps=10,
                max_condition=1e5)
    args.update(kw)
    return BSplineTargets(**args)


def test_bspline_reproduces_spline_and_partition_of_unity():
    fitter = make_fitter(regularization=1e-9)
    controls = torch.randn(2, 5, 10, 3, dtype=torch.float64)
    trace = fitter.decode(controls)
    fitted, valid = fitter.fit(trace, torch.ones(2, 5, 32))
    assert valid.all()
    assert torch.allclose(fitter.decode(fitted.double()), trace, atol=1e-4)
    full_rows = fitter.free_basis.sum(1)
    assert torch.all(full_rows <= 1 + 1e-9) and abs(full_rows[-1].item() - 1) < 1e-9


def test_bspline_stationary_partial_and_invalid():
    fitter = make_fitter()
    trace = torch.zeros(1, 3, 32, 3)
    weight = torch.ones(1, 3, 32)
    trace[0, 1, :, 0] = torch.linspace(0.1, 3.2, 32)
    weight[0, 1, 16:] = 0
    weight[0, 2, 5:] = 0
    controls, valid = fitter.fit(trace, weight)
    assert valid.tolist() == [[True, True, False]]
    assert torch.allclose(controls[0, 0], torch.zeros(10, 3), atol=1e-6)
    assert torch.equal(controls[0, 2], torch.zeros(10, 3))
    decoded = fitter.decode(controls.double())
    assert torch.allclose(decoded[0, 1, :16, 0].float(), trace[0, 1, :16, 0], atol=0.05)


def test_letterbox_roundtrip_and_padding():
    image = np.full((120, 300, 3), 200, dtype=np.uint8)
    canvas, valid, transform = letterbox(image, 224)
    assert canvas.shape == (224, 224, 3) and valid.sum() == 224 * transform.scale_y * 120
    assert not valid[0].any() and valid[112].all()
    xy = np.array([[0.0, 0.0], [300.0, 120.0], [150.0, 60.0]])
    uv = source_xy_to_uv(xy, transform, 224)
    assert np.allclose(uv_to_source_xy(uv, transform, 224), xy)
    assert np.isclose(uv[0, 1] * 224, transform.offset_y)


def test_temporal_plan_respects_gaps():
    timestamps = np.concatenate([np.arange(0, 1.0, 1 / 30), np.arange(1.5, 3.0, 1 / 30)])
    plan = plan_future_samples(timestamps, 0.0, [2 * k / 32 for k in range(1, 33)], max_gap=0.1, tolerance=1e-3)
    times = np.array([2 * k / 32 for k in range(1, 33)])
    expected = (times < 1.0 - 1 / 30 + 1e-9) | (times >= 1.5 - 1e-9)
    assert np.array_equal(plan.valid, expected)
    points = np.repeat(timestamps[None, :, None], 3, axis=2)
    gathered, rel = gather_tracks(points, np.ones((1, len(timestamps))), plan)
    assert np.allclose(gathered[0, plan.valid, 0], times[plan.valid], atol=1e-6)
    assert np.isnan(gathered[0, ~plan.valid]).all() and (rel[0, ~plan.valid] == 0).all()


def test_camera_reference_and_displacement():
    pose = np.eye(4)
    pose[:3, 3] = [0, 0, 2]
    points = np.zeros((2, 4, 3))
    points[0, :, 0] = [0, 1, 2, 3]
    points[1, 2:, 2] = -5
    cam = world_to_reference_camera(points, pose)
    disp, valid, _ = relative_displacements(cam, np.ones((2, 4)), 2.0, 0.5)
    assert np.allclose(disp[0, :, 0], [0.5, 1.0, 1.5]) and valid[0].all()
    assert valid[1].tolist() == [True, False, False] and (disp[1, 1:] == 0).all()
    gl = np.array([[1.0, 2.0, -3.0]])
    assert np.allclose(to_opencv_camera(gl, "opengl"), [[1.0, -2.0, 3.0]])


def test_moving_points_uses_pixel_extent():
    from twe.preprocess.camera_reference import moving_points

    k = np.array([[100.0, 0, 112], [0, 100.0, 112], [0, 0, 1]])
    steps = np.arange(33) / 32
    moving = np.stack([steps * 0.8, np.zeros(33), np.full(33, 2.0)], -1)
    jitter = np.stack([np.sin(steps * 20) * 0.05, np.zeros(33), np.full(33, 2.0)], -1)
    points = np.stack([moving, jitter])
    valid = np.ones((2, 32), bool)
    assert moving_points(points, valid, k, 35.0).tolist() == [True, False]
    valid[0, 4:] = False
    assert moving_points(points, valid, k, 35.0).tolist() == [False, False]


def test_normalizer_uses_only_moving_points(tmp_path):
    from twe.data.synthetic import write_synthetic
    from twe.preprocess.normalizer import load_normalizer

    root = write_synthetic(tmp_path / "data")
    sigma = np.asarray(load_normalizer(root / "normalizer.json")["sigma"])
    assert sigma[:2].max() > 0.05


def test_relabel_moving_rewrites_labels(tmp_path):
    from twe.data.synthetic import write_synthetic
    from twe.preprocess.build_dataset import relabel_moving

    root = write_synthetic(tmp_path / "data")
    loose = relabel_moving(root, 1.0)
    strict = relabel_moving(root, 1e6)
    assert loose > 0 and strict == 0

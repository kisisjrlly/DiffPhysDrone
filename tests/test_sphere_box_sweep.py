"""Continuous sphere geometry, including corner false positives and tunnelling."""
import pytest
import torch

from render.box_scene import swept_box_clearance, swept_box_collision


def clearance(start, end, half=(1., 1., 1.), radius=.12):
    boxes = torch.tensor([[[0., 0., 0., *half]]], dtype=torch.float64)
    return swept_box_clearance(torch.tensor([start], dtype=torch.float64),
                               torch.tensor([end], dtype=torch.float64), boxes, radius).item()


def test_corner_pass_and_exact_tangent():
    assert clearance((1.1, 1.1, -.1), (1.1, 1.1, .1)) == pytest.approx(2**.5 * .1 - .12)
    assert clearance((1.12, -2., 0.), (1.12, 2., 0.)) == pytest.approx(0., abs=1e-12)


def test_stationary_point_inside_interval_not_on_sampling_grid():
    # Closest approach to the corner at t=.375, not an endpoint/slab crossing.
    assert clearance((1.05, 1.20, 0.), (1.25, 1., 0.)) == pytest.approx(.25 / 2**.5 - .12)


def test_thin_wall_crossing_and_zero_displacement():
    assert clearance((-10., 0., 0.), (10., 0., 0.), half=(.0001, 1., 1.)) < 0
    assert clearance((2., 0., 0.), (2., 0., 0.)) == pytest.approx(.88)
    assert clearance((0., 0., 0.), (0., 0., 0.)) == pytest.approx(-1.12)


def test_positive_margin_is_evaluated_continuously():
    assert clearance((1.121, -2., 0.), (1.121, 2., 0.)) < .0011
    assert clearance((1.1212, -2., 0.), (1.1212, 2., 0.)) > .0011


def test_batch_boxes_endpoint_contact_and_finite_gradients():
    boxes = torch.tensor([[[0., 0., 0., 1., 1., 1.]],
                          [[0., 0., 0., 1., 1., 1.]]], dtype=torch.float64)
    start = torch.tensor([[2., 0., 0.], [1.1, 1.1, -.1]], dtype=torch.float64, requires_grad=True)
    end = torch.tensor([[1., 0., 0.], [1.1, 1.1, .1]], dtype=torch.float64, requires_grad=True)
    result = swept_box_clearance(start, end, boxes, .12)
    assert swept_box_collision(start, end, boxes, .12).tolist() == [True, False]
    result.sum().backward()
    assert torch.isfinite(start.grad).all() and torch.isfinite(end.grad).all()
    assert start.grad[1, :2].sum() > 0


def test_exterior_derivatives_match_finite_differences():
    boxes = torch.tensor([[[0., 0., 0., 1., 1., 1.]]], dtype=torch.float64)
    start = torch.tensor([[1.1, 1.3, .1]], dtype=torch.float64, requires_grad=True)
    end = torch.tensor([[1.4, 1.05, .2]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: swept_box_clearance(a, b, boxes, .12), (start, end))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_continuous_clearance_matches_independent_convex_distance_minimization(dtype):
    import numpy as np
    rng = np.random.default_rng(177)
    start, end = rng.uniform(-3,3,(2,128,3))
    half = rng.uniform(.1,1.2,(128,3))
    lo, hi = np.zeros(128), np.ones(128)
    def distance(t):
        return np.linalg.norm(np.maximum(np.abs(start+t[:,None]*(end-start))-half,0),axis=-1)
    # Independent one-dimensional convex search; no slab/candidate equations.
    for _ in range(90):
        a,b = lo+(hi-lo)/3, hi-(hi-lo)/3
        choose_left = distance(a) <= distance(b)
        hi = np.where(choose_left,b,hi)
        lo = np.where(choose_left,lo,a)
    expected = distance((lo+hi)/2)
    boxes = np.concatenate([np.zeros_like(half),half],axis=-1)[:,None]
    result = swept_box_clearance(torch.tensor(start,dtype=dtype),torch.tensor(end,dtype=dtype),
                                 torch.tensor(boxes,dtype=dtype),.12).numpy()
    outside = expected > 1e-8
    np.testing.assert_allclose(result[outside],expected[outside]-.12,atol=2e-6 if dtype==torch.float32 else 1e-10,rtol=1e-6)
    assert np.all(result[~outside] <= -.12+1e-6)

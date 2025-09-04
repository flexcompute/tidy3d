from __future__ import annotations

import autograd.numpy as anp
import numpy as np

import tidy3d as td


class _DummyDerivativeInfo:
    """Minimal stub to drive PolySlab VJP helpers in unit tests.

    - bounds_intersect: ((xmin, ymin, zmin), (xmax, ymax, zmax))
    - adaptive_vjp_spacing: returns dx
    - create_interpolators: unused in these tests
    - evaluate_gradient_at_points: returns g(center) for each patch
    """

    def __init__(self, bounds_intersect, g_fun, dx=0.05):
        self.bounds_intersect = bounds_intersect
        self._g_fun = g_fun
        self._dx = dx

    def adaptive_vjp_spacing(self):
        return self._dx

    def create_interpolators(self, dtype=None):
        return {}

    def evaluate_gradient_at_points(self, centers, normals, perps1, perps2, interpolators):
        return self._g_fun(centers)


def _make_rect(axis=2, theta=0.1, ref_plane="bottom"):
    verts = anp.array([[-0.4, -0.3], [0.4, -0.3], [0.4, 0.3], [-0.4, 0.3]])
    return td.PolySlab(
        vertices=verts,
        slab_bounds=(-0.5, 0.5),
        axis=axis,
        sidewall_angle=theta,
        dilation=0.0,
        reference_plane=ref_plane,
    )


def test_sidewall_constant_g_zero():
    poly = _make_rect(axis=2, theta=0.2)
    # g == 0 everywhere
    di = _DummyDerivativeInfo(
        bounds_intersect=((-1, -1, -1), (1, 1, 1)), g_fun=lambda c: anp.zeros(len(c))
    )
    val = poly._compute_derivative_sidewall_angle(di, anp.array([-1, -1, -1]), anp.array([1, 1, 1]))
    assert np.isclose(val, 0.0)


def test_sidewall_reference_plane_invariance():
    # g ∝ z to avoid zero
    g_fun = lambda centers: centers[:, 2]
    bounds = ((-1, -1, -1), (1, 1, 1))
    di = _DummyDerivativeInfo(bounds_intersect=bounds, g_fun=g_fun)
    vals = []
    for ref in ("bottom", "middle", "top"):
        poly = _make_rect(axis=2, theta=0.2, ref_plane=ref)
        vals.append(
            poly._compute_derivative_sidewall_angle(di, anp.array(bounds[0]), anp.array(bounds[1]))
        )
    assert np.allclose(vals[0], vals[1]) and np.allclose(vals[1], vals[2])


def test_sidewall_orientation_invariance():
    # same polygon, reversed orientation
    g_fun = lambda centers: centers[:, 2]
    di = _DummyDerivativeInfo(bounds_intersect=((-1, -1, -1), (1, 1, 1)), g_fun=g_fun)
    poly = _make_rect(axis=2, theta=0.15)
    val_fwd = poly._compute_derivative_sidewall_angle(
        di, anp.array([-1, -1, -1]), anp.array([1, 1, 1])
    )
    verts_rev = poly.vertices[::-1]
    poly_rev = td.PolySlab(vertices=verts_rev, slab_bounds=(-0.5, 0.5), axis=2, sidewall_angle=0.15)
    val_rev = poly_rev._compute_derivative_sidewall_angle(
        di, anp.array([-1, -1, -1]), anp.array([1, 1, 1])
    )
    assert np.allclose(val_fwd, val_rev)


def test_sidewall_2d_returns_zero():
    poly = _make_rect(axis=2, theta=0.25)
    di = _DummyDerivativeInfo(
        bounds_intersect=((0, 0, 0), (0, 0, 0)), g_fun=lambda c: anp.ones(len(c))
    )
    # Explicitly signal 2D path
    val = poly._compute_derivative_sidewall_angle(
        di, anp.array([-1, -1, -1]), anp.array([1, 1, 1]), is_2d=True
    )
    assert np.isclose(val, 0.0)


def test_sidewall_negative_angle_parity():
    # magnitude should be similar for ±theta for a symmetric g; sign may differ
    g_fun = lambda centers: anp.ones(len(centers))
    di = _DummyDerivativeInfo(bounds_intersect=((-1, -1, -1), (1, 1, 1)), g_fun=g_fun)
    poly_pos = _make_rect(axis=2, theta=0.2)
    poly_neg = _make_rect(axis=2, theta=-0.2)
    v_pos = poly_pos._compute_derivative_sidewall_angle(
        di, anp.array([-1, -1, -1]), anp.array([1, 1, 1])
    )
    v_neg = poly_neg._compute_derivative_sidewall_angle(
        di, anp.array([-1, -1, -1]), anp.array([1, 1, 1])
    )
    assert np.isclose(abs(v_pos), abs(v_neg))


def test_sidewall_xy_clipping():
    # clip half the polygon in x; result should be finite and smaller in magnitude than unclipped
    g_fun = lambda centers: centers[:, 2]
    di_full = _DummyDerivativeInfo(bounds_intersect=((-1, -1, -1), (1, 1, 1)), g_fun=g_fun)
    di_clip = _DummyDerivativeInfo(bounds_intersect=((0, -1, -1), (1, 1, 1)), g_fun=g_fun)
    poly = _make_rect(axis=2, theta=0.2)
    v_full = poly._compute_derivative_sidewall_angle(
        di_full, anp.array([-1, -1, -1]), anp.array([1, 1, 1])
    )
    v_clip = poly._compute_derivative_sidewall_angle(
        di_clip, anp.array([0, -1, -1]), anp.array([1, 1, 1])
    )
    assert np.isfinite(v_clip)
    assert abs(v_clip) <= abs(v_full) + 1e-12

from __future__ import annotations

import numpy as np
import pytest

from tidy3d.components.geometry.polyslab import PolySlab  # adjust import path if needed


# ---- test helpers ----
class FakeDerivativeInfo:
    def __init__(self, g_func, dx):
        self._g = g_func
        self._dx = dx

    def adaptive_vjp_spacing(self, *_, **__):
        return self._dx

    def create_interpolators(self, dtype=None):
        return {}  # unused by our g

    def evaluate_gradient_at_points(self, spatial_coords, normals, perps1, perps2, interpolators):
        return self._g(spatial_coords)


def perimeter_2d(verts):
    v = np.asarray(verts, float)
    e = np.roll(v, -1, axis=0) - v
    return np.linalg.norm(e, axis=1).sum()


def make_square(L=1.0):
    return np.array([[0, 0], [L, 0], [L, L], [0, L]], float)


def expected_square_sidewall_angle_vjp(P, theta, u_min, u_max, *, integrand="constant"):
    """Closed form for a square sidewall using the true offset-polygon perimeter."""
    tan_theta = np.tan(theta)
    inv_cos2 = 1.0 / (np.cos(theta) ** 2)
    if integrand == "constant":
        integral = -P * (u_max**2 - u_min**2) / 2.0
        integral += 8.0 * tan_theta * (u_max**3 - u_min**3) / 3.0
    elif integrand == "linear":
        integral = -P * (u_max**3 - u_min**3) / 3.0
        integral += 8.0 * tan_theta * (u_max**4 - u_min**4) / 4.0
    else:
        raise ValueError(f"Unknown integrand {integrand!r}.")
    return inv_cos2 * integral


# ---- fixtures ----
@pytest.fixture
def geom():
    H = 2.0
    theta = np.deg2rad(15.0)
    verts = make_square(1.0)
    ps = PolySlab(vertices=verts, axis=2, slab_bounds=(-H / 2, H / 2), sidewall_angle=theta)
    P = perimeter_2d(verts)
    return ps, H, theta, P


@pytest.fixture
def sim_bounds():
    return np.array([-10, -10, -10], float), np.array([10, 10, 10], float)


# ---- tests ----


def test_constant_g_matches_closed_form(geom, sim_bounds):
    ps, H, theta, P = geom
    sim_min, sim_max = sim_bounds
    g = lambda xyz: np.ones(xyz.shape[0], dtype=float)
    di = FakeDerivativeInfo(g, dx=H / 50)
    val = ps._compute_derivative_sidewall_angle(
        di, sim_min, sim_max, is_2d=False, interpolators=None
    )
    expected = expected_square_sidewall_angle_vjp(P, theta, -H / 2.0, H / 2.0, integrand="constant")
    assert np.isclose(val, expected, rtol=5e-3, atol=1e-10)


def test_linear_z_matches_closed_form(geom, sim_bounds):
    ps, H, theta, P = geom
    sim_min, sim_max = sim_bounds
    z0 = ps.center_axis
    g = lambda xyz: (xyz[:, 2] - z0)  # g(z)=z_local
    di = FakeDerivativeInfo(g, dx=H / 80)
    val = ps._compute_derivative_sidewall_angle(
        di, sim_min, sim_max, is_2d=False, interpolators=None
    )
    expected = expected_square_sidewall_angle_vjp(P, theta, -H / 2.0, H / 2.0, integrand="linear")
    assert np.isclose(val, expected, rtol=5e-3, atol=1e-10)


def test_2d_returns_zero(geom, sim_bounds):
    ps, H, _theta, _P = geom
    sim_min, sim_max = sim_bounds
    z0 = ps.center_axis
    g = lambda xyz: (xyz[:, 2] - z0)
    di = FakeDerivativeInfo(g, dx=H / 40)
    val = ps._compute_derivative_sidewall_angle(
        di, sim_min, sim_max, is_2d=True, interpolators=None
    )
    assert val == 0.0


def test_vertex_order_invariance(geom, sim_bounds):
    ps, H, _theta, _P = geom
    sim_min, sim_max = sim_bounds
    z0 = ps.center_axis
    g = lambda xyz: (xyz[:, 2] - z0)
    di = FakeDerivativeInfo(g, dx=H / 80)

    val_ccw = ps._compute_derivative_sidewall_angle(
        di, sim_min, sim_max, is_2d=False, interpolators=None
    )

    ps_rev = PolySlab(
        vertices=ps.vertices[::-1],
        axis=2,
        slab_bounds=ps.slab_bounds,
        sidewall_angle=ps.sidewall_angle,
    )
    val_cw = ps_rev._compute_derivative_sidewall_angle(
        di, sim_min, sim_max, is_2d=False, interpolators=None
    )

    assert np.isclose(val_ccw, val_cw, rtol=1e-4, atol=1e-10)


def test_vertex_vjp_order_invariance(geom, sim_bounds):
    ps, H, _theta, _P = geom
    sim_min, sim_max = sim_bounds
    z0 = ps.center_axis
    g = lambda xyz: 1.0 + 0.7 * xyz[:, 0] - 0.2 * xyz[:, 1] + 0.3 * (xyz[:, 2] - z0)
    di = FakeDerivativeInfo(g, dx=H / 80)

    val_ccw = ps._compute_derivative_vertices(di, sim_min, sim_max, is_2d=False, interpolators=None)

    ps_rev = PolySlab(
        vertices=ps.vertices[::-1],
        axis=ps.axis,
        slab_bounds=ps.slab_bounds,
        sidewall_angle=ps.sidewall_angle,
        dilation=ps.dilation,
        reference_plane=ps.reference_plane,
    )
    val_cw = ps_rev._compute_derivative_vertices(
        di, sim_min, sim_max, is_2d=False, interpolators=None
    )

    assert np.allclose(val_ccw, val_cw[::-1], rtol=1e-4, atol=1e-10)


def test_z_clipping_interval(geom):
    ps, H, theta, P = geom
    # clip to upper quarter: [0, H/4] in z_local
    z0 = ps.center_axis
    sim_min = np.array([-10, -10, z0], float)
    sim_max = np.array([10, 10, z0 + H / 4], float)

    g = lambda xyz: (xyz[:, 2] - z0)
    di = FakeDerivativeInfo(g, dx=H / 80)
    val = ps._compute_derivative_sidewall_angle(
        di, sim_min, sim_max, is_2d=False, interpolators=None
    )

    a, b = 0.0, H / 4
    expected = expected_square_sidewall_angle_vjp(P, theta, a, b, integrand="linear")
    assert np.isclose(val, expected, rtol=1e-3, atol=1e-10)


@pytest.mark.parametrize("dx_factor", [1 / 20, 1 / 40, 1 / 80])
def test_convergence_on_dx(geom, sim_bounds, dx_factor):
    ps, H, theta, P = geom
    sim_min, sim_max = sim_bounds
    z0 = ps.center_axis
    g = lambda xyz: (xyz[:, 2] - z0)
    di = FakeDerivativeInfo(g, dx=H * dx_factor)
    val = ps._compute_derivative_sidewall_angle(
        di, sim_min, sim_max, is_2d=False, interpolators=None
    )
    expected = expected_square_sidewall_angle_vjp(P, theta, -H / 2.0, H / 2.0, integrand="linear")
    # allow tighter tolerance as dx shrinks
    tol = {1 / 20: 2e-2, 1 / 40: 1.0e-2, 1 / 80: 5e-3}[dx_factor]
    assert np.isclose(val, expected, rtol=tol, atol=1e-10)


@pytest.mark.parametrize("ref_plane", ["bottom", "middle", "top"])
def test_hinge_reference_plane_constant_g(geom, ref_plane):
    ps, H, theta, P = geom

    # use a slightly reduced thickness to avoid self-intersections for bottom/top hinges
    H_eff = 0.9 * H
    half = 0.5 * H_eff

    ps = PolySlab(
        vertices=ps.vertices,
        axis=ps.axis,
        slab_bounds=(-half, half),
        sidewall_angle=ps.sidewall_angle,
        reference_plane=ref_plane,
    )

    # asymmetric clipping interval relative to center_axis to expose hinge dependence
    z0 = ps.center_axis
    a, b = z0, z0 + H_eff / 4
    sim_min = np.array([-10, -10, a], float)
    sim_max = np.array([10, 10, b], float)

    # constant integrand g(z) = 1
    di = FakeDerivativeInfo(lambda xyz: np.ones(xyz.shape[0], dtype=float), dx=H_eff / 80)
    val = ps._compute_derivative_sidewall_angle(
        di, sim_min, sim_max, is_2d=False, interpolators=None
    )

    z_ref = ps.reference_axis_pos
    expected = expected_square_sidewall_angle_vjp(
        P, theta, a - z_ref, b - z_ref, integrand="constant"
    )
    assert np.isclose(val, expected, rtol=1e-2, atol=1e-10)

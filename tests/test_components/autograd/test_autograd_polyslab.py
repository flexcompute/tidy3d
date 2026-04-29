"""
Tests for ``td.PolySlab._compute_derivatives`` using an integrand of the form

    k(u, v, w) = a u + b v + c w + d

where (u, v) are the in-plane coordinates and w is the extrusion axis.

Test coverage:
 - 3d slabs for each extrusion axis (0, 1, 2)
 - 2d simulation slice (``is_2d``) where the slab is entirely in-plane
 - 2d cross-section where the slab is intersected by a clipping plane
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import shapely.geometry as sgeom
from shapely.geometry.polygon import orient

import tidy3d as td
from tidy3d.components.autograd.derivative_utils import DerivativeInfo

# irregular convex pentagon in the XY-plane
VERTS_2D: np.ndarray = np.array(
    [
        (0.0, 0.0),
        (3.0, 0.0),
        (4.0, 2.0),
        (2.0, 4.0),
        (0.0, 3.0),
    ],
    dtype=float,
)

# slab bounds along the extrusion axis (z-min, z-max)
SLAB_BOUNDS: tuple[float, float] = (0.5, 2.0)

a, b, c, d = 0.3, -0.1, 0.5, 0.8
COEFFS: dict[str, float] = {"a": a, "b": b, "c": c, "d": d}


def polygon_area_centroid(verts: np.ndarray) -> tuple[float, float, float]:
    """Return (area, x dA, y dA) for a CCW polygon (shoelace formulas)"""
    x, y = verts.T
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - y * x1
    area = 0.5 * np.sum(cross)
    mx = np.sum((x + x1) * cross) / 6
    my = np.sum((y + y1) * cross) / 6
    return abs(area), mx, my


def edge_integrals(verts: np.ndarray):
    """Per-edge geometric quantities"""
    v0 = verts
    v1 = np.roll(verts, -1, axis=0)
    seg = v1 - v0

    L = np.linalg.norm(seg, axis=1)  # segment length
    n = np.stack([seg[:, 1], -seg[:, 0]], axis=1) / L[:, None]  # outward normal

    int_x = 0.5 * (v0[:, 0] + v1[:, 0]) * L  # x ds per segment
    int_y = 0.5 * (v0[:, 1] + v1[:, 1]) * L  # y ds per segment

    return L, n, int_x, int_y


def get_edge_intersection(v0, v1, x_clip):
    """Find intersection of edge (v0, v1) with line x=x_clip."""
    x0, y0 = v0
    x1, y1 = v1
    if np.isclose(x0, x1):  # vertical edge
        return None, None
    if (x0 <= x_clip and x1 <= x_clip) or (x0 > x_clip and x1 > x_clip):  # edge doesn't cross
        return None, None

    t = (x_clip - x0) / (x1 - x0)
    if 0 <= t <= 1:
        Iy = y0 + t * (y1 - y0)
        I = np.array([x_clip, Iy])  # noqa: E741
        return I, t
    return None, None


class DummyDI:
    """Stand-in for ``tidy3d.components.autograd.derivative_utils.DerivativeInfo``."""

    def __init__(
        self,
        *,
        paths,
        axis: int,
        coeffs: dict[str, float],
        bounds_intersect: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
        simulation_bounds: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    ) -> None:
        self.paths = paths
        self.axis = axis
        self.a = coeffs["a"]
        self.b = coeffs["b"]
        self.c = coeffs["c"]
        self.d = coeffs["d"]
        self.frequencies = [200e12]
        eps_keys = ["eps_xx", "eps_yy", "eps_zz"]
        self.eps_data = {
            key: td.ScalarFieldDataArray(
                [[[[12.0]]]], coords={"x": [0], "y": [0], "z": [0], "f": [200e12]}
            )
            for key in eps_keys
        }
        self.interpolators = None

        self.bounds_intersect = (
            bounds_intersect
            if bounds_intersect is not None
            else ((-1e3, -1e3, -1e3), (1e3, 1e3, 1e3))
        )

        self.simulation_bounds = (
            simulation_bounds
            if simulation_bounds is not None
            else ((-2e3, -2e3, -2e3), (2e3, 2e3, 2e3))
        )

        self.cached_min_spacing_from_permittivity = None

    min_spacing_from_permittivity = DerivativeInfo.min_spacing_from_permittivity
    adaptive_vjp_spacing = DerivativeInfo.adaptive_vjp_spacing
    wavelength_min = property(lambda self: DerivativeInfo.wavelength_min.fget(self))
    wavelength_max = property(lambda self: DerivativeInfo.wavelength_max.fget(self))

    def create_interpolators(self, dtype=None):
        return {}

    def evaluate_gradient_at_points(
        self, spatial_coords, normals, perps1, perps2, interpolators=None
    ):
        centers = spatial_coords
        plane_axes = [i for i in (0, 1, 2) if i != self.axis]
        u, v = centers[:, plane_axes[0]], centers[:, plane_axes[1]]
        w = centers[:, self.axis]

        vals = self.a * u + self.b * v + self.c * w + self.d
        return vals


def analytic_faces(verts: np.ndarray, coeffs: dict[str, float], z0: float, z1: float):
    """Return the (bottom, top) face forces with our sign convention."""
    area, mx, my = polygon_area_centroid(verts)

    bot = -(coeffs["a"] * mx + coeffs["b"] * my + (coeffs["c"] * z0 + coeffs["d"]) * area)
    top = coeffs["a"] * mx + coeffs["b"] * my + (coeffs["c"] * z1 + coeffs["d"]) * area
    return bot, top


def analytic_vertex_forces(
    verts: np.ndarray, coeffs: dict[str, float], z0: float, z1: float
) -> np.ndarray:
    """
    Return per-vertex forces projected onto the polygon plane,
    using weighted distribution (1-s, s) consistent with linear shape functions.
    """
    a, b, c, d = coeffs["a"], coeffs["b"], coeffs["c"], coeffs["d"]
    thickness = z1 - z0
    n_verts = len(verts)
    vf = np.zeros_like(verts, dtype=float)

    v0 = verts
    v1 = np.roll(verts, -1, axis=0)
    seg = v1 - v0
    L = np.linalg.norm(seg, axis=1)  # segment length (N,)
    L_safe = np.where(L <= np.finfo(float).eps, 1.0, L)

    # ensure normal calculation handles potential zero length segments safely
    n = np.zeros_like(seg)
    valid_edge = L > np.finfo(float).eps
    n[valid_edge] = (
        np.stack([seg[valid_edge, 1], -seg[valid_edge, 0]], axis=1) / L_safe[valid_edge, None]
    )

    x0, y0 = v0[:, 0], v0[:, 1]
    x1, y1 = v1[:, 0], v1[:, 1]

    z_integral_term_half = L * (c / 4.0) * (z1**2 - z0**2)

    # weighted scalar contribution from edge i to vertex i
    F_scalar_i = (
        L * thickness * ((a / 6.0) * (2 * x0 + x1) + (b / 6.0) * (2 * y0 + y1) + d / 2.0)
        + z_integral_term_half
    )

    # weighted scalar contribution from edge 'i' to vertex 'i+1'
    F_scalar_i_plus_1 = (
        L * thickness * ((a / 6.0) * (x0 + 2 * x1) + (b / 6.0) * (y0 + 2 * y1) + d / 2.0)
        + z_integral_term_half
    )

    for i in range(n_verts):
        if not valid_edge[i]:
            continue
        vf[i] += F_scalar_i[i] * n[i]
        vf[(i + 1) % n_verts] += F_scalar_i_plus_1[i] * n[i]

    return vf


@pytest.mark.parametrize("axis", [0, 1, 2])
class TestPolySlab3D:
    """3d test for arbitrary extrusion axis."""

    @pytest.fixture
    def slab(self, axis):
        return td.PolySlab(vertices=VERTS_2D, axis=axis, slab_bounds=SLAB_BOUNDS)

    @pytest.fixture
    def results(self, slab, axis):
        di = DummyDI(
            paths=[("slab_bounds", 0), ("slab_bounds", 1), ("vertices",)],
            axis=axis,
            coeffs=COEFFS,
        )
        return slab._compute_derivatives(di)

    def test_face_forces(self, results):
        bot_exp, top_exp = analytic_faces(VERTS_2D, COEFFS, *SLAB_BOUNDS)
        npt.assert_allclose(results[("slab_bounds", 0)], bot_exp, rtol=1e-2)
        npt.assert_allclose(results[("slab_bounds", 1)], top_exp, rtol=1e-2)

    def test_vertex_forces(self, results):
        vf_exp = analytic_vertex_forces(VERTS_2D, COEFFS, *SLAB_BOUNDS)
        npt.assert_allclose(results[("vertices",)], vf_exp, rtol=1e-2)


class TestPolySlab2DInPlane:
    """Simulation slice exactly through the slab mid-plane (``is_2d == True``)."""

    AXIS = 2

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.slab = td.PolySlab(vertices=VERTS_2D, axis=self.AXIS, slab_bounds=SLAB_BOUNDS)

        z_mid = 0.5 * sum(SLAB_BOUNDS)
        self.di = DummyDI(
            paths=[("slab_bounds", 0), ("slab_bounds", 1), ("vertices",)],
            axis=self.AXIS,
            coeffs=COEFFS,
            bounds_intersect=((-1e3, -1e3, z_mid), (1e3, 1e3, z_mid)),
            simulation_bounds=((-2e3, -2e3, z_mid - 1e3), (2e3, 2e3, z_mid + 1e3)),
        )
        self.results = self.slab._compute_derivatives(self.di)
        self.z_mid = z_mid

    def test_faces_are_zero(self):
        assert self.results[("slab_bounds", 0)] == 0.0
        assert self.results[("slab_bounds", 1)] == 0.0

    def test_vertex_forces(self):
        verts = VERTS_2D
        coeffs = COEFFS
        z_mid = self.z_mid
        a, b, c, d = coeffs["a"], coeffs["b"], coeffs["c"], coeffs["d"]
        n_verts = len(verts)
        vf = np.zeros_like(verts, dtype=float)  # Use float

        v0 = verts
        v1 = np.roll(verts, -1, axis=0)
        seg = v1 - v0
        L = np.linalg.norm(seg, axis=1)
        valid_edge = L > np.finfo(float).eps
        L_safe = np.where(valid_edge, L, 1.0)
        n = np.zeros_like(seg)
        n[valid_edge] = (
            np.stack([seg[valid_edge, 1], -seg[valid_edge, 0]], axis=1) / L_safe[valid_edge, None]
        )

        x0, y0 = v0[:, 0], v0[:, 1]
        x1, y1 = v1[:, 0], v1[:, 1]

        # calculate k = (a*x(s) + b*y(s) + c*z_mid + d)
        k_at_z_mid_common_half = (c * z_mid + d) / 2.0

        # weighted scalar contribution to vertex 'i' (start, weight 1-s)
        F_scalar_i = L * (
            (a / 6.0) * (2 * x0 + x1) + (b / 6.0) * (2 * y0 + y1) + k_at_z_mid_common_half
        )

        # weighted scalar contribution to vertex 'i+1' (end, weight s)
        F_scalar_i_plus_1 = L * (
            (a / 6.0) * (x0 + 2 * x1) + (b / 6.0) * (y0 + 2 * y1) + k_at_z_mid_common_half
        )

        for i in range(n_verts):
            if not valid_edge[i]:
                continue
            vf[i] += F_scalar_i[i] * n[i]
            vf[(i + 1) % n_verts] += F_scalar_i_plus_1[i] * n[i]

        npt.assert_allclose(self.results[("vertices",)], vf, rtol=1e-2)


class TestPolySlab2DCrossSection:
    """Case where the slab surface intersects a vertical clipping plane."""

    AXIS = 2
    X_CLIP = 1.5  # x-coordinate of the clipping plane

    @classmethod
    def setup_class(cls):
        cls.slab = td.PolySlab(vertices=VERTS_2D, axis=cls.AXIS, slab_bounds=SLAB_BOUNDS)

        sim_min = (-1e3, -1e3, -1e3)
        sim_max = (cls.X_CLIP, 1e3, 1e3)

        cls.di = DummyDI(
            paths=[("slab_bounds", 0), ("slab_bounds", 1), ("vertices",)],
            axis=cls.AXIS,
            coeffs=COEFFS,
            bounds_intersect=(sim_min, sim_max),
            simulation_bounds=(sim_min, sim_max),
        )
        cls.results = cls.slab._compute_derivatives(cls.di)

        # intersection geometry: line segments at x == X_CLIP
        poly = sgeom.Polygon(VERTS_2D)
        line = sgeom.LineString([(cls.X_CLIP, -1e3), (cls.X_CLIP, 1e3)])
        inter = poly.intersection(line)

        if inter.is_empty:
            cls.y_segments: list[list[tuple[float, float]]] = []
        elif isinstance(inter, sgeom.MultiLineString):
            cls.y_segments = [list(seg.coords) for seg in inter.geoms]
        else:  # single segment
            cls.y_segments = [list(inter.coords)]

    @classmethod
    def _analytic_face(cls, z) -> float:
        """Exact integral over the clipped face x <= X_CLIP."""
        poly = sgeom.Polygon(VERTS_2D)

        clip = sgeom.box(-1e3, -1e3, cls.X_CLIP, 1e3)
        face = poly.intersection(clip)

        if face.is_empty:
            return 0.0

        face = orient(face, sign=1.0)

        verts = np.asarray(face.exterior.coords[:-1])
        area, mx, my = polygon_area_centroid(verts)

        a_, b_, c_, d_ = (COEFFS[k] for k in ("a", "b", "c", "d"))
        return a_ * mx + b_ * my + (c_ * z + d_) * area

    @staticmethod
    def _integrate_weighted_term(p0, p1, t_start, t_end, weight_s):
        """Integrates ((1-s)p0 + s p1) * weight(s) ds from t_start to t_end."""
        t2_end, t3_end = t_end * t_end, t_end * t_end * t_end
        t2_start, t3_start = t_start * t_start, t_start * t_start * t_start

        if weight_s == "s":
            val_end = p0 * (0.5 * t2_end - t3_end / 3.0) + p1 * (t3_end / 3.0)
            val_start = p0 * (0.5 * t2_start - t3_start / 3.0) + p1 * (t3_start / 3.0)
            return val_end - val_start
        else:  # weight 1-s
            val_end = p0 * (t_end - t2_end + t3_end / 3.0) + p1 * (0.5 * t2_end - t3_end / 3.0)
            val_start = p0 * (t_start - t2_start + t3_start / 3.0) + p1 * (
                0.5 * t2_start - t3_start / 3.0
            )
            return val_end - val_start

    @staticmethod
    def _vertex_forces_clipped(
        original_verts: np.ndarray,
        x_clip: float,
        coeffs: dict[str, float],
        z0: float,
        z1: float,
    ) -> np.ndarray:
        """
        Exact force on each original vertex after clipping ``x <= x_clip``,
        by directly integrating k*N_j over the clipped portion of each original edge.
        """
        a, b, c, d = coeffs["a"], coeffs["b"], coeffs["c"], coeffs["d"]
        thickness = z1 - z0
        n_original_verts = len(original_verts)
        original_vf = np.zeros_like(original_verts, dtype=float)

        v0_all = original_verts
        v1_all = np.roll(original_verts, -1, axis=0)

        for i in range(n_original_verts):
            v0, v1 = v0_all[i], v1_all[i]
            x0, y0 = v0
            x1, y1 = v1

            seg = v1 - v0
            L = np.linalg.norm(seg)
            valid_edge = L > np.finfo(float).eps
            if not valid_edge:
                continue

            n = np.array([seg[1], -seg[0]]) / L  # outward normal

            # determine intersection parameter t (0<=t<=1) for x=x_clip
            t_intersect = 1.0  # full edge potentially inside
            if not np.isclose(x0, x1):
                t_cand = (x_clip - x0) / (x1 - x0)
                if 0 < t_cand < 1:
                    t_intersect = t_cand

            tol = 1e-9
            inside0 = x0 <= x_clip + tol
            inside1 = x1 <= x_clip + tol

            t_start, t_end = 0.0, 1.0

            if not inside0 and not inside1:  # fully outside
                continue
            elif inside0 and not inside1:  # starts inside, ends outside
                t_end = t_intersect
            elif not inside0 and inside1:  # starts outside, ends inside
                t_start = t_intersect

            if t_end <= t_start + tol:
                continue

            t_start = max(0.0, t_start)
            t_end = min(1.0, t_end)
            if t_end <= t_start:
                continue

            # integrate terms weighted by (1-s) for vertex v0 (index i)
            int_s_w0 = (t_end - 0.5 * t_end**2) - (t_start - 0.5 * t_start**2)
            int_x_w0 = TestPolySlab2DCrossSection._integrate_weighted_term(
                x0, x1, t_start, t_end, "1-s"
            )
            int_y_w0 = TestPolySlab2DCrossSection._integrate_weighted_term(
                y0, y1, t_start, t_end, "1-s"
            )

            # integrate terms weighted by s for vertex v1 (index i+1)
            int_s_w1 = (0.5 * t_end**2) - (0.5 * t_start**2)
            int_x_w1 = TestPolySlab2DCrossSection._integrate_weighted_term(
                x0, x1, t_start, t_end, "s"
            )
            int_y_w1 = TestPolySlab2DCrossSection._integrate_weighted_term(
                y0, y1, t_start, t_end, "s"
            )

            # combine with z-integration and coefficients
            z_int_val = 0.5 * (z1**2 - z0**2)

            # scalar force contribution to v0
            F_scalar_i = L * (
                (a * int_x_w0 + b * int_y_w0 + d * int_s_w0) * thickness + c * z_int_val * int_s_w0
            )
            # scalar force contribution to v1
            F_scalar_i_plus_1 = L * (
                (a * int_x_w1 + b * int_y_w1 + d * int_s_w1) * thickness + c * z_int_val * int_s_w1
            )

            original_vf[i] += F_scalar_i * n
            original_vf[(i + 1) % n_original_verts] += F_scalar_i_plus_1 * n

        return original_vf

    def test_vertex_forces(self):
        vf_exp = self._vertex_forces_clipped(VERTS_2D, self.X_CLIP, COEFFS, *SLAB_BOUNDS)
        npt.assert_allclose(self.results[("vertices",)], vf_exp, rtol=1e-2)


class TestPolySlabShortEdges:
    """Test with very short edges to trigger ``n_uniform <= 3`` quadrature path.

    This test uses a small square with edge lengths of 0.05 units. Based on the
    quadrature selection logic in ``PolySlab.get_adaptive_samples()``, edges this short
    should trigger the direct Gauss quadrature path with very few points, as
    ``n_uniform = edge_length / adaptive_vjp_spacing`` will be <= 3.
    """

    AXIS = 2
    # square with very small edge lengths (0.05 units)
    SHORT_EDGE_VERTS = np.array(
        [
            (0.0, 0.0),
            (0.05, 0.0),
            (0.05, 0.05),
            (0.0, 0.05),
        ],
        dtype=float,
    )

    @pytest.fixture
    def slab(self):
        return td.PolySlab(vertices=self.SHORT_EDGE_VERTS, axis=self.AXIS, slab_bounds=SLAB_BOUNDS)

    @pytest.fixture
    def results(self, slab):
        di = DummyDI(
            paths=[("slab_bounds", 0), ("slab_bounds", 1), ("vertices",)],
            axis=self.AXIS,
            coeffs=COEFFS,
        )
        return slab._compute_derivatives(di)

    def test_face_forces(self, results):
        bot_exp, top_exp = analytic_faces(self.SHORT_EDGE_VERTS, COEFFS, *SLAB_BOUNDS)
        npt.assert_allclose(results[("slab_bounds", 0)], bot_exp, rtol=1e-2)
        npt.assert_allclose(results[("slab_bounds", 1)], top_exp, rtol=1e-2)

    def test_vertex_forces(self, results):
        vf_exp = analytic_vertex_forces(self.SHORT_EDGE_VERTS, COEFFS, *SLAB_BOUNDS)
        npt.assert_allclose(results[("vertices",)], vf_exp, rtol=1e-2)


class TestPolySlabLongEdges:
    """Test with very long edges to trigger composite Gauss quadrature (n_gauss > 7).

    This test uses a rectangle with 20-unit long edges. For typical ``adaptive_vjp_spacing``
    values (around 0.1-1.0 wavelength fraction), this results in n_uniform >> 3,
    leading to n_gauss > 7, which triggers the composite Gauss quadrature path that
    breaks the edge into segments with 4-point quadrature each.
    """

    AXIS = 2
    # rectangle with very long edges (20 units)
    LONG_EDGE_VERTS = np.array(
        [
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 1.0),
            (0.0, 1.0),
        ],
        dtype=float,
    )

    @pytest.fixture
    def slab(self):
        return td.PolySlab(vertices=self.LONG_EDGE_VERTS, axis=self.AXIS, slab_bounds=SLAB_BOUNDS)

    @pytest.fixture
    def results(self, slab):
        di = DummyDI(
            paths=[("slab_bounds", 0), ("slab_bounds", 1), ("vertices",)],
            axis=self.AXIS,
            coeffs=COEFFS,
        )
        return slab._compute_derivatives(di)

    def test_face_forces(self, results):
        bot_exp, top_exp = analytic_faces(self.LONG_EDGE_VERTS, COEFFS, *SLAB_BOUNDS)
        npt.assert_allclose(results[("slab_bounds", 0)], bot_exp, rtol=1e-2)
        npt.assert_allclose(results[("slab_bounds", 1)], top_exp, rtol=1e-2)

    def test_vertex_forces(self, results):
        vf_exp = analytic_vertex_forces(self.LONG_EDGE_VERTS, COEFFS, *SLAB_BOUNDS)
        npt.assert_allclose(results[("vertices",)], vf_exp, rtol=1e-2)


class TestPolySlabMixedEdges:
    """Test with mixed edge lengths in the same polygon.

    This test creates a polygon with edges of varying lengths to verify that the
    implementation correctly handles different quadrature methods on different edges
    within the same polygon:
    - Very short edges (0.05 units): Should use direct Gauss quadrature with few points
    - Medium edge (0.95 units): Should use direct Gauss-Legendre quadrature
    - Long edges (10+ units): Should use composite Gauss quadrature

    This tests the robustness of the edge-by-edge quadrature selection.
    """

    AXIS = 2
    # polygon with both very short and very long edges
    MIXED_EDGE_VERTS = np.array(
        [
            (0.0, 0.0),
            (10.0, 0.0),  # long edge: 10 units
            (10.0, 0.05),  # very short edge: 0.05 units
            (10.05, 0.05),  # very short edge: 0.05 units
            (10.05, 1.0),  # medium edge: 0.95 units
            (0.0, 1.0),  # long edge: 10.05 units
        ],
        dtype=float,
    )

    @pytest.fixture
    def slab(self):
        return td.PolySlab(vertices=self.MIXED_EDGE_VERTS, axis=self.AXIS, slab_bounds=SLAB_BOUNDS)

    @pytest.fixture
    def results(self, slab):
        di = DummyDI(
            paths=[("slab_bounds", 0), ("slab_bounds", 1), ("vertices",)],
            axis=self.AXIS,
            coeffs=COEFFS,
        )
        return slab._compute_derivatives(di)

    def test_face_forces(self, results):
        bot_exp, top_exp = analytic_faces(self.MIXED_EDGE_VERTS, COEFFS, *SLAB_BOUNDS)
        npt.assert_allclose(results[("slab_bounds", 0)], bot_exp, rtol=1e-2)
        npt.assert_allclose(results[("slab_bounds", 1)], top_exp, rtol=1e-2)

    def test_vertex_forces(self, results):
        vf_exp = analytic_vertex_forces(self.MIXED_EDGE_VERTS, COEFFS, *SLAB_BOUNDS)
        npt.assert_allclose(results[("vertices",)], vf_exp, rtol=1e-2)


class TestPolySlabDegenerateEdges:
    """Test with near-degenerate edges to verify numerical stability.

    Note: We use small edges (0.001) rather than truly degenerate edges
    (duplicate vertices) because PolySlab validation rejects invalid polygons.
    This tests numerical stability for very small but valid edge lengths.
    """

    AXIS = 2

    @pytest.fixture
    def degenerate_verts(self):
        # pentagon with one small edge to test numerical stability
        return np.array(
            [
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 0.001),  # small edge but above validator tolerance
                (1.0, 1.0),
                (0.0, 1.0),
            ],
            dtype=float,
        )

    @pytest.fixture
    def slab(self, degenerate_verts):
        return td.PolySlab(vertices=degenerate_verts, axis=self.AXIS, slab_bounds=SLAB_BOUNDS)

    @pytest.fixture
    def results(self, slab):
        di = DummyDI(
            paths=[("slab_bounds", 0), ("slab_bounds", 1), ("vertices",)],
            axis=self.AXIS,
            coeffs=COEFFS,
        )
        return slab._compute_derivatives(di)

    def test_derivatives_are_finite(self, results):
        """Ensure derivatives don't have NaN or inf values."""
        for path, value in results.items():
            if isinstance(value, np.ndarray):
                assert np.all(np.isfinite(value)), f"Non-finite values in {path}"
            else:
                assert np.isfinite(value), f"Non-finite value in {path}"

    def test_face_forces(self, results, degenerate_verts):
        # test face forces for polygons with small edges
        bot_exp, top_exp = analytic_faces(degenerate_verts, COEFFS, *SLAB_BOUNDS)
        npt.assert_allclose(results[("slab_bounds", 0)], bot_exp, rtol=1e-2)
        npt.assert_allclose(results[("slab_bounds", 1)], top_exp, rtol=1e-2)


@pytest.mark.parametrize(
    "edge_length,expected_path",
    [
        (0.02, "short"),  # very short edge: n_uniform <= 3
        (0.05, "short"),  # short edge: n_uniform <= 3
        (0.5, "medium"),  # medium edge: direct Gauss quadrature (n_gauss <= 7)
        (2.0, "composite"),  # long edge: composite quadrature (n_gauss > 7)
        (10.0, "composite"),  # long edge: composite quadrature
        (50.0, "composite"),  # very long edge: composite quadrature
    ],
)
class TestQuadraturePaths:
    """Parametrized test to verify correct quadrature path selection based on edge length."""

    AXIS = 2

    @pytest.fixture
    def square_verts(self, edge_length):
        """Create a square with the specified edge length."""
        return np.array(
            [
                (0.0, 0.0),
                (edge_length, 0.0),
                (edge_length, edge_length),
                (0.0, edge_length),
            ],
            dtype=float,
        )

    @pytest.fixture
    def slab(self, square_verts):
        return td.PolySlab(vertices=square_verts, axis=self.AXIS, slab_bounds=SLAB_BOUNDS)

    def test_correct_quadrature_path(self, slab, edge_length, expected_path):
        """Verify that the expected quadrature path is used for the given edge length."""
        di = DummyDI(
            paths=[("vertices",)],
            axis=self.AXIS,
            coeffs=COEFFS,
        )

        # get the adaptive spacing to compute expected path
        # this is not super elegant because it literally copies the implementation
        # but it's better than nothing...
        dx = di.adaptive_vjp_spacing()
        n_uniform = max(1, int(np.ceil(edge_length / dx)))

        if n_uniform <= 3:
            actual_path = "short"
        else:
            n_gauss = max(2, int(n_uniform * 0.4))
            if n_gauss <= 7:
                actual_path = "medium"
            else:
                actual_path = "composite"

        assert actual_path == expected_path, (
            f"Edge length {edge_length} with dx={dx:.6f} gives n_uniform={n_uniform}, "
            f"expected {expected_path} path but got {actual_path}"
        )

        # also verify the derivatives are computed correctly
        results = slab._compute_derivatives(di)
        assert ("vertices",) in results
        assert results[("vertices",)].shape == (4, 2)


def test_composite_quadrature_preserves_target_sample_count():
    """Composite quadrature should segment by target Gauss count, not uniform cells."""
    edge_length = 2.0
    dx = 0.1
    gauss_order = 7
    sample_fraction = 0.4

    n_uniform = int(np.ceil(edge_length / dx))
    n_gauss = max(2, int(n_uniform * sample_fraction))
    expected_segments = int(np.ceil(n_gauss / gauss_order))

    samples, weights = td.PolySlab._adaptive_edge_samples(
        edge_length,
        dx,
        _sample_fraction=sample_fraction,
        _gauss_order=gauss_order,
        _dtype=float,
    )

    assert n_uniform == 20
    assert n_gauss == 8
    assert samples.size == expected_segments * gauss_order
    assert weights.size == expected_segments * gauss_order
    assert np.isclose(np.sum(weights), 1.0)


def test_edge_quadrature_uses_at_least_two_points():
    """Even tiny clipped edges should not collapse to a one-point sidewall quadrature."""
    samples, weights = td.PolySlab._adaptive_edge_samples(
        0.02,
        0.1,
        _sample_fraction=0.4,
        _gauss_order=7,
        _dtype=float,
    )

    assert samples.size == 2
    assert weights.size == 2
    assert np.isclose(np.sum(weights), 1.0)


class TestPolySlabConcave:
    """Test with concave polygons."""

    AXIS = 2
    # L-shaped concave polygon
    CONCAVE_VERTS = np.array(
        [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        ],
        dtype=float,
    )

    @pytest.fixture
    def slab(self):
        return td.PolySlab(vertices=self.CONCAVE_VERTS, axis=self.AXIS, slab_bounds=SLAB_BOUNDS)

    @pytest.fixture
    def results(self, slab):
        di = DummyDI(
            paths=[("slab_bounds", 0), ("slab_bounds", 1), ("vertices",)],
            axis=self.AXIS,
            coeffs=COEFFS,
        )
        return slab._compute_derivatives(di)

    def test_face_forces(self, results):
        bot_exp, top_exp = analytic_faces(self.CONCAVE_VERTS, COEFFS, *SLAB_BOUNDS)
        npt.assert_allclose(results[("slab_bounds", 0)], bot_exp, rtol=1e-2)
        npt.assert_allclose(results[("slab_bounds", 1)], top_exp, rtol=1e-2)

    def test_vertex_forces(self, results):
        vf_exp = analytic_vertex_forces(self.CONCAVE_VERTS, COEFFS, *SLAB_BOUNDS)
        npt.assert_allclose(results[("vertices",)], vf_exp, rtol=1e-2)


@pytest.mark.parametrize("sidewall_angle", [0.0, np.pi / 12, np.pi / 6])
class TestPolySlabSidewallAngles:
    """Test with different sidewall angles.

    Note: We use angles up to pi/6 instead of larger angles like pi/4
    because large sidewall angles with the given slab thickness can cause
    self-intersecting geometry, which fails PolySlab validation.
    """

    AXIS = 2
    TRIANGLE_VERTS = np.array(
        [
            (0.0, 0.0),
            (2.0, 0.0),
            (1.0, 1.5),
        ],
        dtype=float,
    )

    @pytest.fixture
    def slab(self, sidewall_angle):
        return td.PolySlab(
            vertices=self.TRIANGLE_VERTS,
            axis=self.AXIS,
            slab_bounds=SLAB_BOUNDS,
            sidewall_angle=sidewall_angle,
        )

    @pytest.fixture
    def results(self, slab):
        di = DummyDI(
            paths=[("slab_bounds", 0), ("slab_bounds", 1), ("vertices",)],
            axis=self.AXIS,
            coeffs=COEFFS,
        )
        return slab._compute_derivatives(di)

    def test_derivatives_are_finite(self, results):
        """Ensure derivatives are finite for slanted sidewalls."""
        for path, value in results.items():
            if isinstance(value, np.ndarray):
                assert np.all(np.isfinite(value)), f"Non-finite values in {path}"
            else:
                assert np.isfinite(value), f"Non-finite value in {path}"

    def test_face_forces_scale_with_polygon_area(self, slab, results, sidewall_angle):
        """Face forces should scale with the actual polygon areas at each face."""
        base_polygon = slab.base_polygon
        top_polygon = slab.top_polygon

        bot_exp, _ = analytic_faces(base_polygon, COEFFS, *SLAB_BOUNDS)
        _, top_exp = analytic_faces(top_polygon, COEFFS, *SLAB_BOUNDS)

        npt.assert_allclose(results[("slab_bounds", 0)], bot_exp, rtol=1e-2)
        npt.assert_allclose(results[("slab_bounds", 1)], top_exp, rtol=1e-2)

    def test_face_force_area_relationship(self, slab, results, sidewall_angle):
        """Verify that face forces scale linearly with polygon areas."""
        if sidewall_angle == 0.0:
            return

        base_area, _, _ = polygon_area_centroid(slab.base_polygon)
        top_area, _, _ = polygon_area_centroid(slab.top_polygon)
        ref_area, _, _ = polygon_area_centroid(self.TRIANGLE_VERTS)

        # for a linear integrand f(x,y,z) = ax + by + cz + d,
        # the force scales with area when the centroid term dominates
        # this is a sanity check that the implementation is consistent

        # get the reference case (vertical sidewalls)
        slab_vertical = td.PolySlab(
            vertices=self.TRIANGLE_VERTS,
            axis=self.AXIS,
            slab_bounds=SLAB_BOUNDS,
            sidewall_angle=0.0,
        )
        di_vertical = DummyDI(
            paths=[("slab_bounds", 0), ("slab_bounds", 1)],
            axis=self.AXIS,
            coeffs=COEFFS,
        )
        results_vertical = slab_vertical._compute_derivatives(di_vertical)

        # check that the force ratios are reasonable compared to area ratios
        # we don't expect exact proportionality due to centroid shifts,
        # but the relationship should be monotonic and roughly linear
        bot_force_ratio = results[("slab_bounds", 0)] / results_vertical[("slab_bounds", 0)]
        top_force_ratio = results[("slab_bounds", 1)] / results_vertical[("slab_bounds", 1)]

        bot_area_ratio = base_area / ref_area
        top_area_ratio = top_area / ref_area

        # forces should increase/decrease in the same direction as areas
        assert (bot_force_ratio > 1.0) == (bot_area_ratio > 1.0), (
            f"Bottom force ratio {bot_force_ratio} inconsistent with area ratio {bot_area_ratio}"
        )
        assert (top_force_ratio > 1.0) == (top_area_ratio > 1.0), (
            f"Top force ratio {top_force_ratio} inconsistent with area ratio {top_area_ratio}"
        )

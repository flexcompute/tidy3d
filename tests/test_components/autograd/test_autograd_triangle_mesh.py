"""Tests for TriangleMesh autograd derivatives."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

import tidy3d as td
from tidy3d.config import config

from ...utils import AssertLogLevel

VERTICES_TETRA = np.array(
    [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ],
    dtype=float,
)

FACES_TETRA = np.array(
    [
        (0, 2, 1),
        (0, 1, 3),
        (0, 3, 2),
        (1, 2, 3),
    ]
)

VERTICES_OCTA = np.array(
    [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ],
    dtype=float,
)

FACES_OCTA = np.array(
    [
        (4, 0, 2),
        (4, 2, 1),
        (4, 1, 3),
        (4, 3, 0),
        (5, 2, 0),
        (5, 1, 2),
        (5, 3, 1),
        (5, 0, 3),
    ],
    dtype=int,
)

VERTICES_SLENDER_TETRA = np.array(
    [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0e-4, 1.0e-4, 0.0),
        (0.0, 0.0, 1.0),
    ],
    dtype=float,
)

FACES_SLENDER_TETRA = FACES_TETRA

VERTICES_NON_WATERTIGHT = np.array(
    [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ],
    dtype=float,
)

FACES_NON_WATERTIGHT = np.array(
    [
        (0, 1, 2),
        (0, 2, 3),
    ],
    dtype=int,
)

MESH_DEFINITIONS: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "tetrahedron": (VERTICES_TETRA, FACES_TETRA),
    "octahedron": (VERTICES_OCTA, FACES_OCTA),
    "slender_tetrahedron": (VERTICES_SLENDER_TETRA, FACES_SLENDER_TETRA),
}

OUT_OF_BOUNDS_WARNING = "Some triangles from the mesh lie outside the simulation bounds - this may lead to inaccurate gradients."


@pytest.fixture(params=list(MESH_DEFINITIONS.keys()), ids=list(MESH_DEFINITIONS.keys()))
def watertight_mesh(request) -> td.TriangleMesh:
    """Parameterized fixture returning watertight meshes of varying complexity."""

    vertices, faces = MESH_DEFINITIONS[request.param]
    return td.TriangleMesh.from_vertices_faces(vertices, faces)


@pytest.fixture
def non_watertight_mesh() -> td.TriangleMesh:
    """Simple non-watertight surface used to validate graceful handling."""

    return td.TriangleMesh.from_vertices_faces(VERTICES_NON_WATERTIGHT, FACES_NON_WATERTIGHT)


class DummyDerivativeInfo:
    """Lightweight stand-in for ``DerivativeInfo`` used in unit tests."""

    def __init__(
        self,
        grad_func,
        spacing: float = 0.2,
        simulation_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
        | None = None,
    ) -> None:
        self.paths = [("mesh_dataset", "surface_mesh")]
        self.frequencies = [200e12]
        self.eps_in = 12.0
        self.interpolators = {}
        default_bounds = ((-10.0, -10.0, -10.0), (10.0, 10.0, 10.0))
        self.simulation_bounds = simulation_bounds or default_bounds
        self._grad_func = grad_func
        self._spacing = spacing
        self.bounds_intersect = self.simulation_bounds

    def adaptive_vjp_spacing(self) -> float:
        return self._spacing

    def create_interpolators(self, dtype=None):
        return {}

    def evaluate_gradient_at_points(
        self, spatial_coords, normals, perps1, perps2, interpolators=None
    ):
        return self._grad_func(spatial_coords)


def area_and_normal(triangle: np.ndarray) -> tuple[float, np.ndarray]:
    """Return signed area and unit normal for a triangle."""

    edge01 = triangle[1] - triangle[0]
    edge02 = triangle[2] - triangle[0]
    cross = np.cross(edge01, edge02)
    norm = np.linalg.norm(cross)
    if np.isclose(norm, 0.0):
        return 0.0, np.zeros(3, dtype=triangle.dtype)
    return 0.5 * norm, cross / norm


def linear_grad_func_factory(coeffs: np.ndarray, offset: float):
    """Create a linear function g(x) = coeffs.x + offset."""

    def grad_func(points: np.ndarray) -> np.ndarray:
        return points @ coeffs + offset

    return grad_func


def subdivide_triangles(triangles: np.ndarray) -> np.ndarray:
    """Split each triangle into four smaller ones via ``TriangleMesh.subdivide_faces``."""

    triangles = np.asarray(triangles, dtype=float)
    vertices = triangles.reshape(-1, 3)
    face_count = triangles.shape[0]
    faces = np.arange(vertices.shape[0], dtype=int).reshape(face_count, 3)
    refined_vertices, refined_faces = td.TriangleMesh.subdivide_faces(vertices, faces)
    return refined_vertices[refined_faces]


def test_triangle_mesh_gradient_linear_matches_analytic(watertight_mesh):
    """Validate per-vertex gradients against analytic integrals for linear g."""

    mesh = watertight_mesh
    coeffs = np.array([0.6, -0.25, 0.4], dtype=float)
    offset = -0.15
    grad_func = linear_grad_func_factory(coeffs, offset)

    spacing = 0.01 if mesh.triangles.shape[0] <= 4 else 0.005
    derivative_info = DummyDerivativeInfo(grad_func, spacing=spacing)
    grads = mesh._compute_derivatives(derivative_info)[("mesh_dataset", "surface_mesh")]

    expected = np.zeros_like(grads)
    for face_idx, tri in enumerate(mesh.triangles):
        area, normal = area_and_normal(tri)
        if np.isclose(area, 0.0):
            continue

        g_vals = grad_func(tri)
        for local_idx in range(3):
            others = [(local_idx + 1) % 3, (local_idx + 2) % 3]
            gi = g_vals[local_idx]
            gj = g_vals[others[0]]
            gk = g_vals[others[1]]
            integral = area / 12.0 * (2.0 * gi + gj + gk)
            expected[face_idx, local_idx, :] = integral * normal

    # surface integration uses adaptive sampling so allow a few-percent mismatch
    npt.assert_allclose(grads, expected, rtol=8e-2, atol=1e-6)


def test_triangle_mesh_gradient_directional_derivative_matches_quadrature(watertight_mesh):
    """Directional derivative from gradients matches exact integral."""

    mesh = watertight_mesh
    coeffs = np.array([0.3, -0.45, 0.55], dtype=float)
    offset = 0.2
    grad_func = linear_grad_func_factory(coeffs, offset)

    spacing = 0.01 if mesh.triangles.shape[0] <= 4 else 0.005
    derivative_info = DummyDerivativeInfo(grad_func, spacing=spacing)
    grads = mesh._compute_derivatives(derivative_info)[("mesh_dataset", "surface_mesh")]

    rng = np.random.default_rng(1234)
    delta = rng.normal(scale=5e-3, size=grads.shape)

    total_pred = float(np.sum(grads * delta))

    total_exact = 0.0
    weight_matrix = np.full((3, 3), 1.0 / 12.0)
    np.fill_diagonal(weight_matrix, 1.0 / 6.0)

    for face_idx, tri in enumerate(mesh.triangles):
        area, normal = area_and_normal(tri)
        if np.isclose(area, 0.0):
            continue

        g_vals = grad_func(tri)
        dot_vals = delta[face_idx] @ normal
        total_exact += area * dot_vals @ weight_matrix @ g_vals

    npt.assert_allclose(total_pred, total_exact, rtol=1e-3, atol=1e-6)


def test_triangle_mesh_gradient_constant_field_integrates_to_zero(watertight_mesh):
    """Constant surface gradient should integrate to zero net force on a watertight mesh."""

    mesh = watertight_mesh

    def constant_grad(points: np.ndarray) -> np.ndarray:
        return np.ones(points.shape[0], dtype=float)

    derivative_info = DummyDerivativeInfo(constant_grad, spacing=0.01)
    grads = mesh._compute_derivatives(derivative_info)[("mesh_dataset", "surface_mesh")]

    net_force = np.sum(grads, axis=(0, 1))
    npt.assert_allclose(net_force, np.zeros(3), atol=5e-6, rtol=1e-3)


def test_triangle_mesh_gradient_face_permutation_invariant(watertight_mesh):
    """Reordering faces does not change the per-face gradients after reindexing."""

    base_mesh = watertight_mesh
    perm = np.random.default_rng(42).permutation(base_mesh.triangles.shape[0])
    permuted_mesh = td.TriangleMesh.from_triangles(base_mesh.triangles[perm])

    coeffs = np.array([0.2, 0.1, -0.3], dtype=float)
    offset = 0.05
    grad_func = linear_grad_func_factory(coeffs, offset)

    derivative_info = DummyDerivativeInfo(grad_func, spacing=0.01)
    grad_base = base_mesh._compute_derivatives(derivative_info)[("mesh_dataset", "surface_mesh")]
    grad_perm = permuted_mesh._compute_derivatives(derivative_info)[
        ("mesh_dataset", "surface_mesh")
    ]

    inv_perm = np.argsort(perm)
    grad_perm_reordered = grad_perm[inv_perm]

    npt.assert_allclose(grad_base, grad_perm_reordered, rtol=1e-3, atol=1e-6)


def test_triangle_mesh_gradient_zero_when_outside_bounds(watertight_mesh):
    """Gradients vanish when the mesh lies entirely outside the simulation bounds."""

    mesh = watertight_mesh

    def constant_grad(points: np.ndarray) -> np.ndarray:
        return np.ones(points.shape[0], dtype=float)

    far_bounds = ((100.0, 100.0, 100.0), (101.0, 101.0, 101.0))
    derivative_info = DummyDerivativeInfo(
        constant_grad,
        spacing=0.05,
        simulation_bounds=far_bounds,
    )

    grads = mesh._compute_derivatives(derivative_info)[("mesh_dataset", "surface_mesh")]
    npt.assert_allclose(grads, np.zeros_like(grads))


def test_triangle_mesh_gradient_collapsed_axis_samples_intersection_line():
    """Collapsed simulation axes use plane intersections for surface sampling."""

    triangle = np.array([(-0.5, 0.0, 0.0), (0.5, 1.0, 0.0), (0.5, -1.0, 0.0)], dtype=float)
    mesh = td.TriangleMesh.from_triangles(triangle[None, ...])

    def constant_grad(points: np.ndarray) -> np.ndarray:
        return np.ones(points.shape[0], dtype=float)

    bounds = ((0.0, -1.0, -0.1), (0.0, 1.0, 0.1))
    derivative_info = DummyDerivativeInfo(
        constant_grad,
        spacing=0.05,
        simulation_bounds=bounds,
    )

    grads = mesh._compute_derivatives(derivative_info)[("mesh_dataset", "surface_mesh")]

    endpoints = np.array([(0.0, -0.5, 0.0), (0.0, 0.5, 0.0)], dtype=float)
    bary = mesh._barycentric_coordinates(
        triangle,
        endpoints,
        config.adjoint.edge_clip_tolerance,
    )
    length = np.linalg.norm(endpoints[0] - endpoints[1])
    avg_bary = 0.5 * np.sum(bary, axis=0)
    normal = np.array([0.0, 0.0, -1.0])
    expected = (avg_bary[:, None] * length) * normal
    expected = expected.reshape(1, 3, 3)

    npt.assert_allclose(grads, expected, rtol=1e-2, atol=1e-6)


@pytest.mark.parametrize(
    "simulation_bounds, expect_warning",
    [
        (((-0.1, -0.1, -0.1), (0.6, 0.6, 0.6)), True),
        (((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0)), False),
    ],
)
def test_triangle_mesh_partial_bounds_sampling_warns(simulation_bounds, expect_warning):
    """Surface sampling logs a warning only when triangles leave the simulation bounds."""

    vertices, faces = MESH_DEFINITIONS["tetrahedron"]
    mesh = td.TriangleMesh.from_vertices_faces(vertices, faces)
    sim_min, sim_max = simulation_bounds

    log_ctx = (
        AssertLogLevel("WARNING", contains_str=OUT_OF_BOUNDS_WARNING)
        if expect_warning
        else AssertLogLevel(None)
    )

    with log_ctx:
        samples = mesh._collect_surface_samples(
            triangles=mesh.triangles,
            spacing=0.05,
            sim_min=np.array(sim_min),
            sim_max=np.array(sim_max),
        )

    assert set(samples) == {
        "points",
        "normals",
        "perps1",
        "perps2",
        "weights",
        "faces",
        "barycentric",
    }


def test_triangle_mesh_non_watertight_warns_and_computes(non_watertight_mesh, caplog):
    """Non-watertight meshes should warn but still return finite gradients."""

    def constant_grad(points: np.ndarray) -> np.ndarray:
        return np.ones(points.shape[0], dtype=float)

    derivative_info = DummyDerivativeInfo(constant_grad, spacing=0.05)

    with caplog.at_level("WARNING"):
        grads = non_watertight_mesh._compute_derivatives(derivative_info)[
            ("mesh_dataset", "surface_mesh")
        ]

    assert grads.shape == non_watertight_mesh.triangles.shape
    assert np.all(np.isfinite(grads))
    assert not non_watertight_mesh.trimesh.is_watertight


def test_triangle_mesh_gradients_insensitive_to_face_splitting(watertight_mesh):
    """Refining triangles does not change the directional derivative."""

    base_mesh = watertight_mesh
    base_triangles = base_mesh.triangles
    refined_mesh = td.TriangleMesh.from_triangles(subdivide_triangles(base_triangles))

    coeffs = np.array([0.41, -0.33, 0.27], dtype=float)
    offset = 0.12
    grad_func = linear_grad_func_factory(coeffs, offset)

    spacing = 0.006
    derivative_info = DummyDerivativeInfo(grad_func, spacing=spacing)
    derivative_info_refined = DummyDerivativeInfo(grad_func, spacing=spacing)

    grad_base = base_mesh._compute_derivatives(derivative_info)[("mesh_dataset", "surface_mesh")]
    grad_refined = refined_mesh._compute_derivatives(derivative_info_refined)[
        ("mesh_dataset", "surface_mesh")
    ]

    def displacement_field(points: np.ndarray) -> np.ndarray:
        x, y, z = points.T
        return np.column_stack((0.2 * x - 0.1 * y, 0.15 * y + 0.05 * z, -0.25 * z + 0.12 * x))

    disp_base = displacement_field(base_triangles.reshape(-1, 3)).reshape(base_triangles.shape)
    refined_triangles = refined_mesh.triangles
    disp_refined = displacement_field(refined_triangles.reshape(-1, 3)).reshape(
        refined_triangles.shape
    )

    dir_base = float(np.sum(grad_base * disp_base))
    dir_refined = float(np.sum(grad_refined * disp_refined))

    npt.assert_allclose(dir_base, dir_refined, rtol=5e-3, atol=5e-6)

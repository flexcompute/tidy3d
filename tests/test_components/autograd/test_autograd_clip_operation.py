"""Tests for ``ClipOperation`` autograd support."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Callable

import numpy as np
import numpy.testing as npt
import pytest

import tidy3d as td
from tidy3d import TriangleMesh

DEFAULT_SIM_BOUNDS = ((-10.0, -10.0, -10.0), (10.0, 10.0, 10.0))


def _default_gradient(points: np.ndarray) -> np.ndarray:
    """Deterministic gradient profile used in analytical objective evaluations."""
    if points.size == 0:
        return np.zeros(0, dtype=float)
    return points[:, 0] + 0.5 * points[:, 1] - 0.25 * points[:, 2]


class SimpleDerivativeInfo:
    """Lightweight derivative info stub for geometry autograd testing."""

    def __init__(
        self,
        paths: Sequence[tuple],
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
        *,
        gradient_func: Callable[[np.ndarray], np.ndarray] | None = None,
        spacing: float = 0.01,
        simulation_bounds: tuple[tuple[float, float, float], tuple[float, float, float]]
        | None = None,
    ) -> None:
        self.paths = list(paths)
        self.frequencies = [200e12]
        self.wavelength_min = 1.0
        self.eps_in = 1.0
        self.eps_out = 1.0
        self.interpolators = {}
        self._spacing = spacing
        self.gradient_func = gradient_func or _default_gradient
        self.simulation_bounds = simulation_bounds or DEFAULT_SIM_BOUNDS
        self.bounds = bounds if bounds is not None else self.simulation_bounds
        self.bounds_intersect = self.bounds

    def updated_copy(self, **kwargs):
        clone = copy.copy(self)
        for key, value in kwargs.items():
            setattr(clone, key, value)
        return clone

    def adaptive_vjp_spacing(self) -> float:
        return float(self._spacing)

    def create_interpolators(self, dtype=None):
        return {}

    def evaluate_gradient_at_points(
        self, spatial_coords, normals, perps1, perps2, interpolators=None
    ):
        points = np.asarray(spatial_coords, dtype=float)
        if points.size == 0:
            return np.zeros(0, dtype=float)
        return self.gradient_func(points)


def _tetrahedron_mesh(center: Sequence[float], size: Sequence[float]) -> TriangleMesh:
    """Return a watertight tetrahedron mesh centered at ``center`` with ``size`` extents."""

    center = np.asarray(center, dtype=float)
    half = 0.5 * np.asarray(size, dtype=float)
    cx, cy, cz = center
    hx, hy, hz = half
    vertices = np.array(
        [
            (cx + hx, cy + hy, cz + hz),
            (cx + hx, cy - hy, cz - hz),
            (cx - hx, cy + hy, cz - hz),
            (cx - hx, cy - hy, cz + hz),
        ],
        dtype=float,
    )
    triangles = np.array(
        [
            (vertices[0], vertices[1], vertices[2]),
            (vertices[0], vertices[3], vertices[1]),
            (vertices[0], vertices[2], vertices[3]),
            (vertices[1], vertices[3], vertices[2]),
        ],
        dtype=float,
    )
    return TriangleMesh.from_triangles(triangles)


def build_geometry(
    geometry_type: str, center: Sequence[float], size: Sequence[float]
) -> td.Geometry:
    """Return a geometry instance of the requested type."""

    center = tuple(center)
    size = tuple(size)
    if geometry_type == "sphere":
        radius = 0.5 * min(size)
        return td.Sphere(center=center, radius=radius)
    if geometry_type == "box":
        return td.Box(center=center, size=size)

    if geometry_type == "polyslab":
        half_x = size[0] / 2.0
        half_y = size[1] / 2.0
        half_z = size[2] / 2.0
        vertices = np.array(
            [
                (center[0] - half_x, center[1] - half_y),
                (center[0] + half_x, center[1] - half_y),
                (center[0], center[1] + half_y),
            ],
            dtype=float,
        )
        slab_bounds = (center[2] - half_z, center[2] + half_z)
        return td.PolySlab(vertices=vertices, axis=2, slab_bounds=slab_bounds)

    if geometry_type == "mesh":
        return _tetrahedron_mesh(center=center, size=size)

    raise ValueError(f"Unsupported geometry type '{geometry_type}'.")


SAMPLE_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (1.5, 0.0, 0.0),
        (-0.5, 0.0, 0.0),
    ],
    dtype=float,
)
SAMPLE_NORMALS = np.array(
    [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ],
    dtype=float,
)
SAMPLE_PERPS1 = np.array(
    [
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
    ],
    dtype=float,
)
SAMPLE_PERPS2 = np.array(
    [
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ],
    dtype=float,
)
SAMPLE_WEIGHTS = np.ones(3, dtype=float)
SAMPLE_FACES = np.zeros(3, dtype=int)
SAMPLE_BARY = np.full((3, 3), 1.0 / 3.0, dtype=float)
INSIDE_MASK = np.array([True, False, True], dtype=bool)


def _sample_dict() -> dict[str, np.ndarray]:
    """Return a fresh copy of the mocked sampling dictionary."""

    return {
        "points": SAMPLE_POINTS.copy(),
        "normals": SAMPLE_NORMALS.copy(),
        "perps1": SAMPLE_PERPS1.copy(),
        "perps2": SAMPLE_PERPS2.copy(),
        "weights": SAMPLE_WEIGHTS.copy(),
        "faces": SAMPLE_FACES.copy(),
        "barycentric": SAMPLE_BARY.copy(),
    }


def _make_mesh() -> td.TriangleMesh:
    """Create a simple triangle mesh for testing."""
    vertices = np.array(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ],
        dtype=float,
    )
    faces = np.array([(0, 1, 2)], dtype=int)
    return td.TriangleMesh.from_vertices_faces(vertices, faces)


def _patch_sample_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace surface sampling with deterministic data."""

    def fake_collect(self, *args, **kwargs):
        return _sample_dict()

    monkeypatch.setattr(td.TriangleMesh, "_collect_surface_samples", fake_collect, raising=True)


class _BaseDerivativeInfo:
    """Shared helpers for derivative info stubs."""

    def adaptive_vjp_spacing(self) -> float:
        return 0.5

    def create_interpolators(self, dtype=None):
        return {}


class MinimalDerivativeInfo(_BaseDerivativeInfo):
    """Lightweight derivative info container for ClipOperation routing tests."""

    def __init__(self, paths) -> None:
        self.paths = [tuple(path) for path in paths]
        self.interpolators = None
        self.bounds = ((-2.0, -2.0, -2.0), (2.0, 2.0, 2.0))
        self.simulation_bounds = self.bounds
        self.bounds_intersect = self.bounds
        self.frequencies = [200e12]
        self.E_der_map = {}
        self.D_der_map = {}
        self.E_fwd = {}
        self.E_adj = {}
        self.D_fwd = {}
        self.D_adj = {}
        self.eps_data = {}
        self.eps_in = 1.0
        self.eps_out = 1.0
        self.wavelength_min = 1.0

    def updated_copy(self, **kwargs):
        kwargs.pop("deep", None)
        kwargs.pop("validate", None)
        new = MinimalDerivativeInfo(self.paths)
        new.__dict__.update(self.__dict__)
        if "paths" in kwargs:
            new.paths = list(kwargs.pop("paths"))
        for key, value in kwargs.items():
            setattr(new, key, value)
        return new

    def evaluate_gradient_at_points(
        self,
        spatial_coords,
        normals,
        perps1,
        perps2,
        interpolators=None,
    ):
        return np.zeros(len(spatial_coords), dtype=float)


class RecordingDerivativeInfo(_BaseDerivativeInfo):
    """DerivativeInfo stub that records sampling points."""

    def __init__(self) -> None:
        self.paths = [("mesh_dataset", "surface_mesh")]
        self.simulation_bounds = ((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))
        self.bounds_intersect = self.simulation_bounds
        self.interpolators: dict | None = {}
        self.last_points: np.ndarray | None = None
        self.last_normals: np.ndarray | None = None

    def evaluate_gradient_at_points(
        self,
        spatial_coords,
        normals,
        perps1,
        perps2,
        interpolators=None,
    ):
        self.last_points = np.array(spatial_coords)
        self.last_normals = np.array(normals)
        return np.ones(spatial_coords.shape[0], dtype=float)


class RecordingDerivativeInfoNested(MinimalDerivativeInfo):
    """DerivativeInfo that records points for nested ClipOperation tests."""

    def __init__(self, paths) -> None:
        super().__init__(paths)
        self.last_points: np.ndarray | None = None
        self.last_normals: np.ndarray | None = None

    def updated_copy(self, **kwargs):
        kwargs.pop("deep", None)
        kwargs.pop("validate", None)
        if "paths" in kwargs:
            self.paths = list(kwargs.pop("paths"))
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self

    def evaluate_gradient_at_points(
        self,
        spatial_coords,
        normals,
        perps1,
        perps2,
        interpolators=None,
    ):
        self.last_points = np.asarray(spatial_coords, dtype=float)
        self.last_normals = np.asarray(normals, dtype=float)
        return np.ones(self.last_points.shape[0], dtype=float)


@pytest.mark.parametrize(
    ("operation", "expected_use", "expected_flip"),
    [
        ("intersection", INSIDE_MASK, np.array([False, False, False], dtype=bool)),
        ("union", ~INSIDE_MASK, np.array([False, False, False], dtype=bool)),
        ("difference", ~INSIDE_MASK, np.array([False, False, False], dtype=bool)),
        ("symmetric_difference", np.array([True, True, True], dtype=bool), INSIDE_MASK),
    ],
)
def test_triangle_mesh_clip_filters_geometry_a(operation, expected_use, expected_flip, monkeypatch):
    """TriangleMesh sampling honors ClipOperation rules for geometry_a."""

    mesh = _make_mesh()
    _patch_sample_collection(monkeypatch)
    info = RecordingDerivativeInfo()
    other = td.Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
    clip = td.ClipOperation(operation=operation, geometry_a=mesh, geometry_b=other)

    mesh._compute_derivatives(info, clip_operation=(clip, "geometry_a"))

    expected_points = SAMPLE_POINTS[expected_use]
    expected_normals = SAMPLE_NORMALS.copy()
    expected_normals[expected_flip] *= -1.0
    npt.assert_allclose(info.last_points, expected_points)
    npt.assert_allclose(info.last_normals, expected_normals[expected_use])


@pytest.mark.parametrize(
    ("operation", "expected_use", "expected_flip"),
    [
        ("intersection", INSIDE_MASK, np.array([False, False, False], dtype=bool)),
        ("union", ~INSIDE_MASK, np.array([False, False, False], dtype=bool)),
        ("difference", INSIDE_MASK, INSIDE_MASK),
        ("symmetric_difference", np.array([True, True, True], dtype=bool), INSIDE_MASK),
    ],
)
def test_triangle_mesh_clip_filters_geometry_b(operation, expected_use, expected_flip, monkeypatch):
    """TriangleMesh sampling honors ClipOperation rules for geometry_b."""

    mesh = _make_mesh()
    _patch_sample_collection(monkeypatch)
    info = RecordingDerivativeInfo()
    other = td.Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
    clip = td.ClipOperation(operation=operation, geometry_a=other, geometry_b=mesh)

    mesh._compute_derivatives(info, clip_operation=(clip, "geometry_b"))

    expected_points = SAMPLE_POINTS[expected_use]
    expected_normals = SAMPLE_NORMALS.copy()
    expected_normals[expected_flip] *= -1.0
    npt.assert_allclose(info.last_points, expected_points)
    npt.assert_allclose(info.last_normals, expected_normals[expected_use])


@pytest.mark.parametrize(
    "geometry_type, paths, expected_contexts",
    [
        (
            "box",
            [("geometry_a", "center", 0), ("geometry_b", "size", 1)],
            ("geometry_a", "geometry_b"),
        ),
        ("polyslab", [("geometry_a", "vertices")], ("geometry_a",)),
        ("sphere", [("geometry_a", "radius")], ("geometry_a",)),
    ],
)
def test_clip_operation_passes_clip_context(
    geometry_type: str,
    paths: list[tuple],
    expected_contexts: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
):
    """``ClipOperation`` forwards context when differentiating geometries."""

    contexts: list[tuple[td.ClipOperation, str] | None] = []

    def fake_mesh_derivatives(self, derivative_info, clip_operation=None):
        contexts.append(clip_operation)
        triangles = np.asarray(self.triangles, dtype=float)
        return {("mesh_dataset", "surface_mesh"): np.zeros_like(triangles)}

    monkeypatch.setattr(
        td.TriangleMesh, "_compute_derivatives", fake_mesh_derivatives, raising=True
    )

    if geometry_type == "polyslab":
        vertices = np.array(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=float)
        geometry_a = td.PolySlab(vertices=vertices, slab_bounds=(-0.5, 0.5), axis=2)
        geometry_b = td.Box(center=(0.0, 0.0, 0.0), size=(3.0, 3.0, 3.0))
    elif geometry_type == "sphere":
        geometry_a = td.Sphere(center=(0.0, 0.0, 0.0), radius=0.6)
        geometry_b = td.Box(center=(0.0, 0.0, 0.0), size=(3.0, 3.0, 3.0))
    else:
        geometry_a = td.Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
        geometry_b = td.Box(center=(1.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))

    clip = td.ClipOperation(operation="union", geometry_a=geometry_a, geometry_b=geometry_b)
    info = MinimalDerivativeInfo(paths=paths)

    result = clip._compute_derivatives(info)

    expected = [(clip, which) for which in expected_contexts]
    assert contexts == expected
    for path in paths:
        assert path in result


def test_nested_clip_operation_derivatives():
    """Nested ``ClipOperation`` should support mesh-based derivatives."""

    inner_a = td.Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
    inner_b = td.Box(center=(0.4, 0.0, 0.0), size=(0.8, 0.8, 0.8))
    inner_clip = td.ClipOperation(operation="difference", geometry_a=inner_a, geometry_b=inner_b)
    outer_b = td.Box(center=(0.0, 0.0, 0.0), size=(2.0, 2.0, 2.0))
    outer_clip = td.ClipOperation(operation="union", geometry_a=inner_clip, geometry_b=outer_b)

    paths = [
        ("geometry_a", "geometry_a", "center", 0),
        ("geometry_a", "geometry_b", "size", 1),
        ("geometry_b", "size", 2),
    ]
    info = MinimalDerivativeInfo(paths=paths)

    result = outer_clip._compute_derivatives(info)

    for path in paths:
        assert path in result


def test_nested_clip_operation_applies_outer_clip(monkeypatch):
    """Nested ClipOperation should apply outer clip context to inner geometry samples."""

    _patch_sample_collection(monkeypatch)
    mesh = _make_mesh()

    inner_far_box = td.Box(center=(10.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
    inner_clip = td.ClipOperation(operation="union", geometry_a=mesh, geometry_b=inner_far_box)

    outer_box = td.Box(center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0))
    outer_clip = td.ClipOperation(
        operation="intersection", geometry_a=inner_clip, geometry_b=outer_box
    )

    paths = [("geometry_a", "geometry_a", "mesh_dataset", "surface_mesh")]
    info = RecordingDerivativeInfoNested(paths=paths)

    outer_clip._compute_derivatives(info)

    expected_mask = outer_box.inside(SAMPLE_POINTS[:, 0], SAMPLE_POINTS[:, 1], SAMPLE_POINTS[:, 2])
    expected_points = SAMPLE_POINTS[expected_mask]
    expected_normals = SAMPLE_NORMALS[expected_mask]

    npt.assert_allclose(info.last_points, expected_points)
    npt.assert_allclose(info.last_normals, expected_normals)


FIELD_PATHS = {
    "sphere": ("radius",),
    "box": ("center", 0),
    "polyslab": ("slab_bounds", 0),
    "mesh": ("mesh_dataset", "surface_mesh"),
}


@pytest.mark.parametrize("geometry_type", ["sphere", "box", "polyslab", "mesh"])
@pytest.mark.parametrize(
    "operation, expected_scales",
    [
        ("union", (1.0, 0.0)),
        ("intersection", (0.0, 1.0)),
        ("difference", (1.0, -1.0)),
        ("symmetric_difference", (1.0, -1.0)),
    ],
)
def test_clip_operation_known_gradient_relations(geometry_type, operation, expected_scales):
    """Compare ClipOperation gradients against analytical expectations for nested boxes."""

    center_a = (0.0, 0.0, 0.0)
    center_b = (0.2, 0.0, 0.0)
    size_a = (2.0, 1.4, 1.0)
    size_b = (1.0, 0.8, 0.6)

    geometry_a = build_geometry(geometry_type, center=center_a, size=size_a)
    geometry_b = build_geometry(geometry_type, center=center_b, size=size_b)

    field_path = FIELD_PATHS[geometry_type]
    geo_di_a = SimpleDerivativeInfo(paths=[field_path], bounds=geometry_a.bounds)
    geo_di_b = SimpleDerivativeInfo(paths=[field_path], bounds=geometry_b.bounds)
    if geometry_type == "sphere":
        baseline_a = geometry_a._compute_derivatives_via_mesh(geo_di_a).get(field_path, 0.0)
        baseline_b = geometry_b._compute_derivatives_via_mesh(geo_di_b).get(field_path, 0.0)
    else:
        baseline_a = geometry_a._compute_derivatives(geo_di_a).get(field_path, 0.0)
        baseline_b = geometry_b._compute_derivatives(geo_di_b).get(field_path, 0.0)

    clip = td.ClipOperation(operation=operation, geometry_a=geometry_a, geometry_b=geometry_b)
    clip_path_a = ("geometry_a", *field_path)
    clip_path_b = ("geometry_b", *field_path)
    clip_di = SimpleDerivativeInfo(paths=[clip_path_a, clip_path_b], bounds=clip.bounds)
    gradients = clip._compute_derivatives(clip_di)

    expected_scale_a, expected_scale_b = expected_scales
    grad_a = gradients.get(clip_path_a, 0.0)
    grad_b = gradients.get(clip_path_b, 0.0)

    rtol = 6e-2
    atol = 3e-2

    np.testing.assert_allclose(
        np.asarray(grad_a, dtype=float),
        expected_scale_a * np.asarray(baseline_a, dtype=float),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        np.asarray(grad_b, dtype=float),
        expected_scale_b * np.asarray(baseline_b, dtype=float),
        rtol=rtol,
        atol=atol,
    )

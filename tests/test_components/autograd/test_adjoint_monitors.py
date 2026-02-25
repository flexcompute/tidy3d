"""Tests for adjoint monitor sizing on planar simulations."""

from __future__ import annotations

import numpy as np
import pytest

import tidy3d as td

SIM_FIELDS_KEYS = [("structures", 0, "geometry")]

POLY_VERTS_2D: np.ndarray = np.array(
    [
        (0.0, 0.0),
        (3.0, 0.0),
        (4.0, 2.0),
        (2.0, 4.0),
        (0.0, 3.0),
    ],
    dtype=float,
)


def _make_2d_simulation(structure: td.Structure) -> td.Simulation:
    return td.Simulation(
        size=(4.0, 4.0, 0.0),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.2),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=False),
        structures=[structure],
        sources=[],
        monitors=[
            td.FieldMonitor(center=(0, 0, 0), size=(0, 0, 0), freqs=[2e14], name="ref"),
        ],
    )


def _make_tetra_mesh() -> td.TriangleMesh:
    # Reuse the same tetra mesh everywhere (matches the earlier 2D test geometry).
    vertices = np.array(
        [
            (1.0, 0.0, -0.1),
            (-1.0, 0.0, 0.1),
            (0.0, 1.0, 0.1),
            (0.0, -1.0, 0.1),
        ],
        dtype=float,
    )
    faces = np.array(
        [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
        ],
        dtype=int,
    )
    return td.TriangleMesh.from_vertices_faces(vertices, faces)


@pytest.mark.parametrize(
    "center_z, expected_size",
    [
        (0.0, (1.0, 1.0, 0.0)),
        (0.25, (2 * np.sqrt(0.5**2 - 0.25**2),) * 2 + (0.0,)),
    ],
)
def test_adjoint_monitors_use_plane_bounds_sphere(center_z, expected_size):
    structure = td.Structure(
        geometry=td.Sphere(radius=0.5, center=(0, 0, center_z)), medium=td.Medium()
    )
    sim = _make_2d_simulation(structure)

    monitors_field, monitors_eps = sim._make_adjoint_monitors(SIM_FIELDS_KEYS)

    assert monitors_field[0].size == pytest.approx(expected_size)
    assert monitors_field[0].center == pytest.approx((0.0, 0.0, 0.0))
    assert monitors_eps[0].size == pytest.approx(expected_size)


def test_adjoint_monitors_use_plane_bounds_mesh():
    mesh = _make_tetra_mesh()
    structure = td.Structure(geometry=mesh, medium=td.Medium())
    sim = _make_2d_simulation(structure)

    monitors_field, monitors_eps = sim._make_adjoint_monitors(SIM_FIELDS_KEYS)

    assert monitors_field[0].size == pytest.approx((0.5, 1.0, 0.0))
    assert monitors_field[0].center == pytest.approx((0.25, 0.0, 0.0))
    assert monitors_eps[0].size == pytest.approx((0.5, 1.0, 0.0))


def test_adjoint_monitors_use_plane_bounds_mesh_disjoint_components():
    """
    Disjoint mesh components: adjoint-plane monitor should use the union of
    all intersection bounds (not just one component).
    """

    # Two identical tetrahedra, separated in x, symmetric about the origin.
    # Each component spans:
    #   x: [-2, -1] and [1, 2]
    #   y: [-1,  1] for both
    vertices = np.array(
        [
            # Left component (x in [-2, -1])
            (-2.0, 0.0, 1.0),  # 0
            (-1.0, 0.0, -1.0),  # 1
            (-2.0, 1.0, -1.0),  # 2
            (-2.0, -1.0, -1.0),  # 3
            # Right component (x in [1, 2])
            (2.0, 0.0, 1.0),  # 4
            (1.0, 0.0, -1.0),  # 5
            (2.0, 1.0, -1.0),  # 6
            (2.0, -1.0, -1.0),  # 7
        ],
        dtype=float,
    )

    # Faces for each tetrahedron (same connectivity, offset by +4 for right)
    faces = np.array(
        [
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
            (4, 5, 6),
            (4, 5, 7),
            (4, 6, 7),
            (5, 6, 7),
        ],
        dtype=int,
    )

    mesh = td.TriangleMesh.from_vertices_faces(vertices, faces)
    structure = td.Structure(geometry=mesh, medium=td.Medium())
    sim = _make_2d_simulation(structure)

    monitors_field, monitors_eps = sim._make_adjoint_monitors(SIM_FIELDS_KEYS)

    # Union across both components:
    # x spans [-2, 2] -> size 4
    # y spans [-0.5, 0.5] -> size 1 (note we are interested in z=0 plane, mid y between +-1 and 0)
    # z size is 0 for a 2D plane monitor
    assert monitors_field[0].size == pytest.approx((4.0, 1.0, 0.0))
    assert monitors_field[0].center == pytest.approx((0.0, 0.0, 0.0))
    assert monitors_eps[0].size == pytest.approx((4.0, 1.0, 0.0))


def _make_3d_simulation(structure: td.Structure) -> td.Simulation:
    return td.Simulation(
        size=(4.0, 4.0, 4.0),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.2),
        boundary_spec=td.BoundarySpec.pml(x=True, y=True, z=True),
        structures=[structure],
        sources=[],
        monitors=[
            td.FieldMonitor(center=(0, 0, 0), size=(0, 0, 0), freqs=[2e14], name="ref"),
        ],
    )


@pytest.mark.parametrize(
    "geometry",
    [
        td.Sphere(radius=0.5, center=(0.3, -0.2, 0.1)),
        td.Cylinder(radius=0.3, length=0.8, center=(-0.5, 0.4, -0.1), axis=2),
        td.Box(center=(0.2, 0.1, -0.3), size=(0.6, 0.8, 0.4)),
        _make_tetra_mesh(),
        td.PolySlab(vertices=POLY_VERTS_2D, axis=2, slab_bounds=(-1, 1)),
    ],
    ids=["sphere", "cylinder", "box", "mesh", "polyslab"],
)
def test_adjoint_monitors_3d_use_geometry_bounding_box(geometry):
    structure = td.Structure(geometry=geometry, medium=td.Medium())
    sim = _make_3d_simulation(structure)

    monitors_field, monitors_eps = sim._make_adjoint_monitors(SIM_FIELDS_KEYS)

    expected_box = geometry.bounding_box

    assert monitors_field[0].size == pytest.approx(tuple(expected_box.size))
    assert monitors_field[0].center == pytest.approx(tuple(expected_box.center))
    assert monitors_eps[0].size == pytest.approx(tuple(expected_box.size))
    assert monitors_eps[0].center == pytest.approx(tuple(expected_box.center))

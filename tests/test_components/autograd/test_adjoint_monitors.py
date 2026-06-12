"""Tests for adjoint monitor sizing on planar simulations."""

from __future__ import annotations

import numpy as np
import pydantic as pd
import pytest

import tidy3d as td
from tidy3d.components.autograd.flux_monitor import is_flux_adjoint_helper_name
from tidy3d.components.structure import _expand_adjoint_monitor_box

from ...utils import assert_single_value_error_loc

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
    expected_box = _expand_adjoint_monitor_box(
        td.Box(center=(0.0, 0.0, 0.0), size=expected_size), sim.grid
    )

    assert monitors_field[0].size == pytest.approx(tuple(expected_box.size))
    assert monitors_field[0].center == pytest.approx((0.0, 0.0, 0.0))
    assert monitors_eps[0].size == pytest.approx(tuple(expected_box.size))


def test_adjoint_monitors_use_plane_bounds_mesh():
    mesh = _make_tetra_mesh()
    structure = td.Structure(geometry=mesh, medium=td.Medium())
    sim = _make_2d_simulation(structure)

    monitors_field, monitors_eps = sim._make_adjoint_monitors(SIM_FIELDS_KEYS)
    expected_box = _expand_adjoint_monitor_box(
        td.Box(center=(0.25, 0.0, 0.0), size=(0.5, 1.0, 0.0)), sim.grid
    )

    assert monitors_field[0].size == pytest.approx(tuple(expected_box.size))
    assert monitors_field[0].center == pytest.approx(tuple(expected_box.center))
    assert monitors_eps[0].size == pytest.approx(tuple(expected_box.size))


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
    expected_box = _expand_adjoint_monitor_box(
        td.Box(center=(0.0, 0.0, 0.0), size=(4.0, 1.0, 0.0)), sim.grid
    )

    # Union across both components:
    # x spans [-2, 2] -> size 4
    # y spans [-0.5, 0.5] -> size 1 (note we are interested in z=0 plane, mid y between +-1 and 0)
    # z size is 0 for a 2D plane monitor
    assert monitors_field[0].size == pytest.approx(tuple(expected_box.size))
    assert monitors_field[0].center == pytest.approx(tuple(expected_box.center))
    assert monitors_eps[0].size == pytest.approx(tuple(expected_box.size))


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

    expected_box = _expand_adjoint_monitor_box(geometry.bounding_box, sim.grid)

    assert monitors_field[0].size == pytest.approx(tuple(expected_box.size))
    assert monitors_field[0].center == pytest.approx(tuple(expected_box.center))
    assert monitors_eps[0].size == pytest.approx(tuple(expected_box.size))
    assert monitors_eps[0].center == pytest.approx(tuple(expected_box.center))


@pytest.mark.parametrize("use_colocated_integration", [True, False])
def test_flux_monitor_adjoint_helpers_are_internal_and_opt_in(use_colocated_integration):
    structure = td.Structure(geometry=td.Box(size=(1.0, 1.0, 1.0)), medium=td.Medium())
    flux_monitor = td.FluxMonitor(
        center=(0, 0, 0),
        size=(1, 1, 1),
        freqs=[2e14],
        name="flux",
        exclude_surfaces=("z-",),
        apodization=td.ApodizationSpec(start=1e-15, end=2e-15, width=1e-16),
        enable_adjoint=True,
        use_colocated_integration=use_colocated_integration,
    )
    sim = _make_3d_simulation(structure).updated_copy(monitors=(flux_monitor,))

    sim_with_helpers = sim._with_adjoint_monitors(SIM_FIELDS_KEYS)
    helper_monitors = [
        monitor
        for monitor in sim_with_helpers.monitors
        if is_flux_adjoint_helper_name(monitor.name)
    ]

    assert len(helper_monitors) == 5
    assert all(isinstance(monitor, td.FieldMonitor) for monitor in helper_monitors)
    assert all(monitor.apodization == flux_monitor.apodization for monitor in helper_monitors)
    # helpers follow the parent's integration scheme so the differentiated frontend flux
    # functional matches the stored solver flux
    assert all(monitor.colocate == use_colocated_integration for monitor in helper_monitors)
    assert all(
        monitor.use_colocated_integration == use_colocated_integration
        for monitor in helper_monitors
    )
    for surface, helper_monitor in zip(
        flux_monitor.integration_surfaces, helper_monitors, strict=True
    ):
        assert helper_monitor.fields == td.FluxMonitor._adjoint_tangential_field_components(surface)

    monitors_fld, monitors_eps = sim._make_adjoint_monitors(SIM_FIELDS_KEYS)
    sim_without_helpers = sim.updated_copy(monitors=tuple(monitors_fld + monitors_eps))
    assert not any(
        is_flux_adjoint_helper_name(monitor.name) for monitor in sim_without_helpers.monitors
    )

    sim_untracked = sim.updated_copy(monitors=(flux_monitor.updated_copy(enable_adjoint=False),))
    assert sim_untracked._freqs_adjoint == []
    assert not any(
        is_flux_adjoint_helper_name(monitor.name)
        for monitor in sim_untracked._with_adjoint_monitors([]).monitors
    )


def test_flux_monitor_enable_adjoint_rejects_empty_surface_set():
    """Tracked box FluxMonitors need at least one hidden field-helper surface."""
    with pytest.raises(
        pd.ValidationError, match=r"enable_adjoint=True.*no integration surfaces"
    ) as excinfo:
        td.FluxMonitor(
            center=(0, 0, 0),
            size=(1, 1, 1),
            freqs=[2e14],
            name="flux",
            exclude_surfaces=("x-", "x+", "y-", "y+", "z-", "z+"),
            enable_adjoint=True,
        )
    assert_single_value_error_loc(
        excinfo,
        ("exclude_surfaces",),
        message_contains="enable_adjoint=True",
    )


def test_flux_monitor_helper_validation_label_uses_parent_name(monkeypatch):
    """Hidden helper validation messages should point to the user FluxMonitor."""
    flux_monitor = td.FluxMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=[2e14],
        name="flux",
        enable_adjoint=True,
    )
    sim = td.Simulation(
        size=(2, 2, 2),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.5),
        monitors=(flux_monitor,),
    )
    sim_with_helpers = sim._with_adjoint_monitors([])
    helper_monitor = next(
        monitor
        for monitor in sim_with_helpers.monitors
        if is_flux_adjoint_helper_name(monitor.name)
    )

    label = sim_with_helpers._monitor_validation_label(helper_monitor)
    assert label == "hidden adjoint field helper for FluxMonitor 'flux'"
    assert "__tidy3d_flux_adjoint" not in label
    assert (
        sim_with_helpers._monitor_validation_index(
            monitor_name=helper_monitor.name, fallback_index=1
        )
        == 0
    )

    monkeypatch.setattr(td.FieldMonitor, "_storage_size_solver", lambda *_, **__: 100e9)
    with pytest.raises(pd.ValidationError) as excinfo:
        sim_with_helpers._validate_monitor_size()
    assert_single_value_error_loc(
        excinfo,
        ("monitors", 0),
        message_contains="hidden adjoint field helper for FluxMonitor 'flux'",
    )


def test_flux_monitor_helper_name_collision_uses_standard_duplicate_name_error():
    """Flux helper names follow the existing adjoint-monitor reserved-name convention."""
    flux_monitor = td.FluxMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=[2e14],
        name="flux",
        enable_adjoint=True,
    )
    colliding_user_monitor = td.FieldMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=[2e14],
        name="__tidy3d_flux_adjoint_0_0",
    )
    sim = td.Simulation(
        size=(2, 2, 2),
        run_time=1e-12,
        grid_spec=td.GridSpec.uniform(dl=0.5),
        monitors=(flux_monitor, colliding_user_monitor),
    )

    with pytest.raises(pd.ValidationError, match=r"'monitors' names are not unique"):
        sim._with_adjoint_monitors([])


def test_enable_adjoint_is_flux_monitor_only():
    """The autograd storage flag should not leak to DirectivityMonitor."""
    assert "enable_adjoint" in td.FluxMonitor.model_fields
    assert "enable_adjoint" not in td.DirectivityMonitor.model_fields
    assert td.DirectivityMonitor.enable_adjoint is False

    flux_monitor = td.FluxMonitor(
        center=(0, 0, 0),
        size=(1, 1, 0),
        freqs=[2e14],
        name="flux",
    )
    assert flux_monitor.model_dump()["enable_adjoint"] is False
    assert '"enable_adjoint":false' in flux_monitor.model_dump_json()
    sim_with_default_flux = _make_2d_simulation(
        td.Structure(geometry=td.Box(size=(1, 1, 0)), medium=td.Medium())
    ).updated_copy(monitors=(flux_monitor,))
    assert '"enable_adjoint":false' in sim_with_default_flux.model_dump_json()

    flux_monitor_opt_in = flux_monitor.updated_copy(enable_adjoint=True)
    sim_with_opt_in_flux = sim_with_default_flux.updated_copy(monitors=(flux_monitor_opt_in,))
    assert flux_monitor_opt_in.model_dump()["enable_adjoint"] is True
    assert '"enable_adjoint":true' in flux_monitor_opt_in.model_dump_json()
    assert '"enable_adjoint":true' in sim_with_opt_in_flux.model_dump_json()
    assert sim_with_default_flux._hash_self() != sim_with_opt_in_flux._hash_self()

    monitor = td.DirectivityMonitor(
        center=(0, 0, 0),
        size=(1, 1, 1),
        freqs=[2e14],
        name="directivity",
        theta=[0],
        phi=[0],
    )
    assert "enable_adjoint" not in monitor.model_dump()

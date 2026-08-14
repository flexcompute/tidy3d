"""Geometry helpers shared by mode solving and FDTD simulation validation."""

from __future__ import annotations

from math import isclose
from typing import TYPE_CHECKING, Any

import numpy as np

from tidy3d.components.geometry.base import Box
from tidy3d.components.geometry.utils import (
    SnapBehavior,
    SnapLocation,
    SnappingSpec,
    snap_box_to_grid,
)
from tidy3d.components.material.tensor_rotation import bend_axis_global_axis
from tidy3d.components.source.field import ModeSource

if TYPE_CHECKING:
    from tidy3d.components.grid.grid import Grid
    from tidy3d.components.monitor import AbstractModeMonitor
    from tidy3d.components.types import Axis, FreqArray
    from tidy3d.components.types.mode_spec import ModeSpecType


def effective_mode_plane(plane: Box, simulation_geometry: Box) -> Box:
    """Return the portion of a mode plane inside the simulation geometry."""
    mode_plane_bounds = Box.bounds_intersection(plane.bounds, simulation_geometry.bounds)
    return Box.from_bounds(*mode_plane_bounds)


def snapped_mode_domain(grid: Grid, box: Box, normal_axis: int) -> Box:
    """Snap a mode plane outward to grid boundaries along its tangential axes."""
    behavior = [SnapBehavior.Off] * 3
    location = [SnapLocation.Boundary] * 3
    for axis in range(3):
        if axis != normal_axis and grid.num_cells[axis] > 1:
            behavior[axis] = SnapBehavior.Expand
    snap_spec = SnappingSpec(location=tuple(location), behavior=tuple(behavior))
    return snap_box_to_grid(grid, box, snap_spec)


def bend_axis_3d_from_plane_and_mode_spec(plane: Box, mode_spec: ModeSpecType) -> Axis:
    """Return the global 3D bend axis for a mode plane and mode specification."""
    normal_axis = plane.size.index(0.0)
    if mode_spec.bend_axis is not None:
        return bend_axis_global_axis(normal_axis=normal_axis, bend_axis=mode_spec.bend_axis)
    _, plane_axes = plane.pop_axis((0, 1, 2), axis=normal_axis)
    rotation_axis_index = int(abs(np.cos(mode_spec.angle_phi)))
    return plane_axes[rotation_axis_index]


def rotation_translate_kwargs(plane: Box, mode_spec: ModeSpecType) -> dict[str, float]:
    """Return translations applied around an angled mode-plane rotation."""
    bend_axis_3d = bend_axis_3d_from_plane_and_mode_spec(plane, mode_spec)
    _, (idx_u, idx_v) = plane.pop_axis((0, 1, 2), axis=bend_axis_3d)
    translate_coords = [0.0, 0.0, 0.0]
    translate_coords[idx_u] = plane.center[idx_u]
    translate_coords[idx_v] = plane.center[idx_v]
    return dict(zip("xyz", translate_coords))


def rotation_kwargs(plane: Box, mode_spec: ModeSpecType) -> dict[str, float | Axis]:
    """Return the rotation applied to structures intersecting an angled mode plane."""
    normal_axis = plane.size.index(0.0)
    bend_axis_3d = bend_axis_3d_from_plane_and_mode_spec(plane, mode_spec)
    angle_theta = mode_spec.angle_theta
    angle_phi = mode_spec.angle_phi
    theta_map = {
        (0, 2): -angle_theta * np.cos(angle_phi),
        (0, 1): angle_theta * np.sin(angle_phi),
        (1, 2): angle_theta * np.cos(angle_phi),
        (1, 0): -angle_theta * np.sin(angle_phi),
        (2, 1): -angle_theta * np.cos(angle_phi),
        (2, 0): angle_theta * np.sin(angle_phi),
    }
    return {"angle": theta_map.get((normal_axis, bend_axis_3d), 0.0), "axis": bend_axis_3d}


def rotation_validation_freqs(mode_object: ModeSource | AbstractModeMonitor) -> FreqArray:
    """Return the frequencies relevant to validating angled structure rotations."""
    if isinstance(mode_object, ModeSource):
        freqs = np.asarray(mode_object.frequency_grid, dtype=float)
    else:
        freqs = np.asarray(mode_object.freqs, dtype=float)
    return np.asarray(mode_object.mode_spec._sampling_freqs_mode_solver(freqs=freqs), dtype=float)


def solver_symmetry(simulation: Any, plane: Box) -> tuple[int, int]:
    """Return simulation symmetry expressed in the two mode-plane axes."""
    normal_axis = plane.size.index(0.0)
    mode_symmetry = list(simulation.symmetry)
    for dim in range(3):
        if not isclose(simulation.center[dim], plane.center[dim]):
            mode_symmetry[dim] = 0
    _, symmetry = plane.pop_axis(mode_symmetry, axis=normal_axis)
    return tuple(symmetry)


def mode_plane_grid(simulation: Any, plane: Box) -> tuple[np.ndarray, np.ndarray]:
    """Return the two boundary-coordinate arrays used by a mode plane."""
    normal_axis = plane.size.index(0.0)
    _, tangential_axes = plane.pop_axis([0, 1, 2], normal_axis)

    span_inds = simulation._discretize_inds_monitor(plane, colocate=False)
    symmetry = solver_symmetry(simulation, plane)
    for dim, num_cells in enumerate(simulation.grid.num_cells):
        if num_cells <= 1:
            span_inds[dim] = [0, 1]
    for dim, value in enumerate(symmetry):
        if value != 0:
            axis = tangential_axes[dim]
            span_inds[axis, 0] += np.diff(span_inds[axis])[0] // 2

    mode_grid = simulation._subgrid(span_inds=span_inds)
    grid_snapped = mode_grid.snap_to_box_zero_dim(plane)
    grid_snapped = simulation._snap_zero_dim(grid_snapped, skip_axis=normal_axis)
    boundaries = grid_snapped.boundaries.to_list
    return boundaries[tangential_axes[0]], boundaries[tangential_axes[1]]

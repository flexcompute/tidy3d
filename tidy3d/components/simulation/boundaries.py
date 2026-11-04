"""Boundary validation, PML geometry, and symmetry helpers for Yee-grid simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np

from tidy3d.components.base import cached_property
from tidy3d.components.boundary import (
    PML,
    ABCBoundary,
    Absorber,
    AbsorberSpec,
    BlochBoundary,
    Boundary,
    ModeABCBoundary,
    Periodic,
    StablePML,
)
from tidy3d.components.geometry.base import Box
from tidy3d.components.grid.snapping import (
    _shift_object,
)
from tidy3d.log import log

if TYPE_CHECKING:
    from tidy3d.components.boundary import InternalAbsorber
    from tidy3d.components.types import Axis

from . import constants


def normalize_boundaries_for_zero_dims(sim: Any, warn_on_change: bool = True) -> None:
    """Replace the absorbing boundary on both sides of a zero-size axis with ``Periodic``.

    Only an axis whose two sides are the same absorber changes, so a boundary that the rule
    ``validate_boundaries_for_zero_dims`` rejects keeps the sides it reports.
    """
    val = sim.boundary_spec
    for dim, (boundary, size_dim) in enumerate(zip(val.to_list, sim.size)):
        if (
            size_dim != 0
            or boundary[0] != boundary[1]
            or not isinstance(boundary[0], AbsorberSpec | ABCBoundary | ModeABCBoundary)
        ):
            continue
        axis = "xyz"[dim]
        val = val.updated_copy(**{axis: Boundary(minus=Periodic(), plus=Periodic())})
        if warn_on_change:
            log.warning(
                f"The simulation has zero size along the {axis} axis, so "
                "using a PML or absorbing boundary along that axis is incorrect. "
                f"Use either 'Periodic' or 'BlochBoundary' along {axis}. "
                "Using 'Periodic' boundary by default."
            )
    if val != sim.boundary_spec:
        object.__setattr__(sim, "boundary_spec", val)


@cached_property
def _shifted_internal_absorbers(self: Any) -> list[InternalAbsorber]:
    """List of absorber shifted to their actual locations based on their grid_shift's."""

    return [
        _shift_object(
            obj=absorber,
            grid=self.grid,
            bounds=self.bounds,
            direction=absorber.direction,
            shift=absorber.grid_shift,
        )
        for absorber in self.internal_absorbers
    ]


@cached_property
def bounds_pml(self: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Simulation bounds including the PML regions."""
    log.warning(
        "'Simulation.bounds_pml' will be removed in Tidy3D 3.0. "
        "Use 'Simulation.simulation_bounds' instead."
    )
    return self.simulation_bounds


@cached_property
def simulation_bounds(self: Any) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Simulation bounds including the PML regions."""
    pml_thick = self.pml_thicknesses
    bounds_in = self.bounds
    bounds_min = tuple((bmin - pml[0] for bmin, pml in zip(bounds_in[0], pml_thick)))
    bounds_max = tuple((bmax + pml[1] for bmax, pml in zip(bounds_in[1], pml_thick)))

    return (bounds_min, bounds_max)


def _make_pml_boxes(self: Any, normal_axis: Axis) -> list[Box]:
    """make a list of Box objects representing the pml to plot on plane."""
    pml_boxes = []
    pml_thicks = self.pml_thicknesses
    for pml_axis, num_layers_dim in enumerate(self.num_pml_layers):
        if pml_axis == normal_axis:
            continue
        for sign, pml_height, num_layers in zip((-1, 1), pml_thicks[pml_axis], num_layers_dim):
            if num_layers == 0:
                continue
            pml_box = self._make_pml_box(pml_axis=pml_axis, pml_height=pml_height, sign=sign)
            pml_boxes.append(pml_box)
    return pml_boxes


def _make_pml_box(self: Any, pml_axis: Axis, pml_height: float, sign: int) -> Box:
    """Construct a :class:`.Box` representing an arborbing boundary to be plotted."""
    rmin, rmax = (list(bounds) for bounds in self.simulation_bounds)
    if sign == -1:
        rmax[pml_axis] = rmin[pml_axis] + pml_height
    else:
        rmin[pml_axis] = rmax[pml_axis] - pml_height
    pml_box = Box.from_bounds(rmin=rmin, rmax=rmax)

    # if any dimension of the sim has size 0, set the PML to a very small size along that dim
    new_size = list(pml_box.size)
    for dim_index, sim_size in enumerate(self.size):
        if sim_size == 0.0:
            new_size[dim_index] = constants.PML_HEIGHT_FOR_0_DIMS
    pml_box = pml_box.updated_copy(size=tuple(new_size))

    return pml_box


@cached_property
def pml_thicknesses(self: Any) -> list[tuple[float, float]]:
    """Thicknesses (um) of absorbers in all three axes and directions (-, +)

    Returns
    -------
    list[Tuple[float, float]]
        List containing the absorber thickness (micron) in - and + boundaries.
    """
    num_layers = self.num_pml_layers
    pml_thicknesses = []
    for num_layer, boundaries in zip(num_layers, self.grid.boundaries.to_list):
        thick_l = boundaries[num_layer[0]] - boundaries[0]
        thick_r = boundaries[-1] - boundaries[-1 - num_layer[1]]
        pml_thicknesses.append((thick_l, thick_r))

    return pml_thicknesses


@cached_property
def _periodic(self: Any) -> tuple[bool, bool, bool]:
    """For each dimension, ``True`` if periodic/Bloch boundaries and ``False`` otherwise.
    We check on both sides but in practice there should be no cases in which a periodic/Bloch
    BC is on one side only. This is explicitly validated for Bloch, and implicitly done for
    periodic, in which case we allow PEC/PMC on the other side, but we replace the periodic
    boundary with another PEC/PMC plane upon initialization."""
    periodic = []
    for bcs_1d in self.boundary_spec.to_list:
        periodic.append(all(isinstance(bcs, Periodic | BlochBoundary) for bcs in bcs_1d))
    return periodic


@cached_property
def num_pml_layers(self: Any) -> list[tuple[float, float]]:
    """Number of absorbing layers in all three axes and directions (-, +).

    Returns
    -------
    list[tuple[float, float]]
        List containing the number of absorber layers in - and + boundaries.
    """
    num_layers = [[0, 0], [0, 0], [0, 0]]

    for idx_i, boundary1d in enumerate(self.boundary_spec.to_list):
        for idx_j, boundary in enumerate(boundary1d):
            if isinstance(boundary, PML | StablePML | Absorber):
                num_layers[idx_i][idx_j] = boundary.num_layers

    return num_layers


def _num_non_pml_cells(self: Any) -> int:
    """Number of grid cells in the simulation domain excluding PML/absorber layers."""
    non_pml_cells_dim = []
    for num_cells_dim, num_pml_layers_dim in zip(self.grid.num_cells, self.num_pml_layers):
        num_pml_cells_dim = num_pml_layers_dim[0] + num_pml_layers_dim[1]
        non_pml_cells_dim.append(num_cells_dim - num_pml_cells_dim)
    return int(np.prod(non_pml_cells_dim))

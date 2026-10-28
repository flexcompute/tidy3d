"""Boundary validation, PML geometry, and symmetry helpers for Yee-grid simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from pydantic import (
    model_validator,
)

from tidy3d._runtime import always_validate
from tidy3d.components.base import cached_property
from tidy3d.components.boundary import (
    CLIPPING_MARGIN,
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
from tidy3d.components.source.field import AbstractModeSource
from tidy3d.components.source.frame import PECFrame
from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Callable

    from tidy3d.compat import Self
    from tidy3d.components.boundary import BoundaryEdgeType, BoundarySpec, InternalAbsorber
    from tidy3d.components.types import Axis

from . import constants


def validate_boundaries_for_zero_dims(
    warn_on_change: bool = True,
) -> Callable[[Any], Any]:
    """Error if absorbing boundaries, bloch boundaries, unmatching pec/pmc, or symmetry is used along a zero dimension."""

    @model_validator(mode="after")
    @always_validate
    def boundaries_for_zero_dims(self: Any) -> Any:
        """Error if absorbing boundaries, bloch boundaries, unmatching pec/pmc, or symmetry is used along a zero dimension."""
        val = self.boundary_spec
        boundaries = val.to_list
        size = self.size
        symmetry = self.symmetry
        axis_names = "xyz"

        for dim, (boundary, symmetry_dim, size_dim) in enumerate(zip(boundaries, symmetry, size)):
            if size_dim == 0:
                axis = axis_names[dim]
                num_absorbing_bdries = sum(
                    isinstance(bnd, AbsorberSpec | ABCBoundary | ModeABCBoundary)
                    for bnd in boundary
                )
                num_bloch_bdries = sum(isinstance(bnd, BlochBoundary) for bnd in boundary)

                if num_absorbing_bdries > 0:
                    pbc = Boundary(minus=Periodic(), plus=Periodic())
                    val = val.updated_copy(**{axis: pbc})
                    if warn_on_change:
                        log.warning(
                            f"The simulation has zero size along the {axis} axis, so "
                            "using a PML or absorbing boundary along that axis is incorrect. "
                            f"Use either 'Periodic' or 'BlochBoundary' along {axis}. "
                            "Using 'Periodic' boundary by default."
                        )

                if num_bloch_bdries > 0:
                    self._raise_validation_error_at_loc(
                        f"The simulation has zero size along the {axis} axis, "
                        "using a Bloch boundary along such an axis is not supported because of "
                        "the Bloch vector definition in units of '2 * pi / (size along dimension)'. Use a small "
                        "but nonzero size along the dimension instead.",
                        "boundary_spec",
                        axis,
                    )

                if symmetry_dim != 0:
                    self._raise_validation_error_at_loc(
                        f"The simulation has zero size along the {axis} axis, so "
                        "using symmetry along that axis is incorrect. Use 'PECBoundary' "
                        "or 'PMCBoundary' to select source polarization if needed and set "
                        f"Simulation.symmetry to 0 along {axis}.",
                        "symmetry",
                        dim,
                    )

                if boundary[0] != boundary[1]:
                    self._raise_validation_error_at_loc(
                        f"The simulation has zero size along the {axis} axis. "
                        f"The boundary condition for {axis} plus and {axis} "
                        "minus must be the same.",
                        "boundary_spec",
                        axis,
                    )

        # Update boundary_spec if it was modified
        if val != self.boundary_spec:
            object.__setattr__(self, "boundary_spec", val)

        return self

    return boundaries_for_zero_dims


def _validate_boundary_spec_symmetry(self: Any) -> Self:
    """Error if symmetry is imposed along an axis but the boundary conditions are not the same
    on both sides."""

    def equivalent(plus: BoundarySpec, minus: BoundarySpec) -> bool:
        """Returns whether two boundary conditions are physically identical."""
        # Make copies of `plus` and `minus` with the `name` attribute set to "".
        plus_cpy = plus.updated_copy(name="")
        minus_cpy = minus.updated_copy(name="")
        return plus_cpy == minus_cpy

    bs = self.boundary_spec
    boundaries = [bs.x, bs.y, bs.z]
    for ax, symmetry, ax_bounds in zip("xyz", self.symmetry, boundaries):
        if symmetry != 0 and not equivalent(ax_bounds.plus, ax_bounds.minus):
            self._raise_validation_error_at_loc(
                f"Symmetry '{symmetry}' along axis {ax} requires the same boundary "
                f"condition on both sides of the axis.",
                "boundary_spec",
                ax,
            )
    return self


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


def _pml_extrusion_clipping_bound_ind(self: Any, axis: int, side: int) -> int | None:
    """Grid-boundary index where the PML extrusion clipping inset reaches into the domain.

    The "extrusion region" spans from the outer simulation boundary through the PML and an
    additional :data:`CLIPPING_MARGIN` cells of interior. This method returns the inner-most
    boundary index of that region (i.e., ``num_layers + CLIPPING_MARGIN`` on the minus side
    and the analogous index counting from the far end on the plus side).

    Returns ``None`` when there are no absorber layers on that side or the computed index
    falls outside the grid. This helper does not check whether ``extrude_structures`` is
    actually enabled — callers that only care when extrusion is active must gate separately.
    """
    n_layers = self.num_pml_layers[axis][side]
    if n_layers == 0:
        return None
    n_bounds = len(self.grid.boundaries.to_list[axis])
    if side == 0:
        idx = n_layers + CLIPPING_MARGIN
        return idx if idx < n_bounds else None
    idx = n_bounds - 1 - n_layers - CLIPPING_MARGIN
    return idx if idx >= 0 else None


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


def _validate_frequency_mode_abc(self: Any) -> Self:
    """Warn if ModeABCBoundary expects a frequency from a source, but there are multiple sources with different central frequencies."""

    def boundary_needs_freq(
        boundary: ModeABCBoundary | ABCBoundary | BoundaryEdgeType,
    ) -> bool:
        return (isinstance(boundary, ModeABCBoundary) and boundary.freq_spec is None) or (
            isinstance(boundary, ABCBoundary)
            and (
                (boundary.conductivity is not None and boundary.conductivity != 0)
                or (boundary.permittivity is None and boundary.conductivity is None)
            )
        )

    # check domain boundaries
    boundaries = self.boundary_spec.to_list
    need_wavelength = any(boundary_needs_freq(edge) for edge in np.ravel(boundaries))

    # check dinternal absorbers
    need_wavelength = need_wavelength or any(
        boundary_needs_freq(abc.boundary_spec) for abc in self.internal_absorbers
    )

    if need_wavelength:
        self._check_source_freq_available(
            no_source_error=(
                "At least one 'ModeABCBoundary'/'ABCBoundary' needs specification of frequency at which the absorbed mode must be evaluated. "
                "Add at least one source or use parameter 'frequency' for 'ModeABCBoundary'."
            ),
            no_source_loc=("sources",),
            multi_freq_warning=(
                "At least one 'ModeABCBoundary' does not specify frequency at which the absorbed mode must be evaluated. "
                "The central frequency of the first source will be used."
            ),
        )

    return self


def _validate_absorber_in_zero_dims(self: Any) -> Self:
    """Error if internal absorber is oriented along zero size dim."""
    val = self.internal_absorbers
    if val is None:
        return val

    sim_size = self.size
    for abc_index, abc in enumerate(val):
        if sim_size[abc._normal_axis] == 0:
            self._raise_validation_error_at_loc(
                "Port absorbers are not allowed to be oriented along simulation zero size dimensions.",
                "internal_absorbers",
                abc_index,
            )

    return self


def _validate_pec_frame_not_in_pml_extrusion(self: Any) -> None:
    """Error if an automatically added PEC frame overlaps the PML extrusion region.

    Works in grid-boundary index space: each PEC frame spans ``[beg, end]`` along every axis,
    and each PML side with ``extrude_structures`` enabled forbids the index range covering
    the PML plus an additional ``CLIPPING_MARGIN`` cells of interior (the clipping inset).
    Touching counts as overlap.
    """
    # Collect auto-added PEC frame index spans alongside the field/loc they originate from.
    frames: list[tuple[np.ndarray, str, int, str]] = []
    for src_idx, src in enumerate(self.sources):
        if isinstance(src, AbstractModeSource) and isinstance(src.frame, PECFrame):
            span_inds, _, _ = self._pec_frame_span_inds(src)
            descr = f"mode source '{src.name}'" if src.name else f"mode source at index {src_idx}"
            frames.append((span_inds, "sources", src_idx, descr))
    for abs_idx, absorber in enumerate(self._shifted_internal_absorbers):
        span_inds, _, _ = self._pec_frame_span_inds(absorber)
        frames.append(
            (span_inds, "internal_absorbers", abs_idx, f"internal absorber at index {abs_idx}")
        )
    if not frames:
        return

    boundaries = self.boundary_spec.to_list
    grid_boundaries = self.grid.boundaries.to_list

    for axis in range(3):
        n_bounds = len(grid_boundaries[axis])
        for side in (0, 1):
            bnd = boundaries[axis][side]
            if not isinstance(bnd, AbsorberSpec) or not bnd.extrude_structures:
                continue
            clip_ind = self._pml_extrusion_clipping_bound_ind(axis, side)
            if clip_ind is None:
                continue
            ext_lo, ext_hi = (0, clip_ind) if side == 0 else (clip_ind, n_bounds - 1)
            for span_inds, field, loc, descr in frames:
                beg, end = span_inds[axis]
                if beg <= ext_hi and end >= ext_lo:
                    axis_label = "xyz"[axis]
                    side_label = f"{'-+'[side]}{axis_label}"
                    self._raise_validation_error_at_loc(
                        f"The automatically added PEC frame for {descr} overlaps the "
                        f"{bnd.type} extrusion region on the '{side_label}' boundary. "
                        f"The extrusion region extends {CLIPPING_MARGIN} grid cells beyond "
                        f"the {bnd.type} into the simulation domain; increase the simulation "
                        f"size along '{axis_label}', move the source/absorber away from "
                        f"that boundary, or disable 'extrude_structures' on that side.",
                        field,
                        loc,
                    )


def _num_non_pml_cells(self: Any) -> int:
    """Number of grid cells in the simulation domain excluding PML/absorber layers."""
    non_pml_cells_dim = []
    for num_cells_dim, num_pml_layers_dim in zip(self.grid.num_cells, self.num_pml_layers):
        num_pml_cells_dim = num_pml_layers_dim[0] + num_pml_layers_dim[1]
        non_pml_cells_dim.append(num_cells_dim - num_pml_cells_dim)
    return int(np.prod(non_pml_cells_dim))

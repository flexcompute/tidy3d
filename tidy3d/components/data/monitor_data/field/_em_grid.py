from __future__ import annotations

from math import isclose
from typing import TYPE_CHECKING

import autograd.numpy as np

from tidy3d.components.data.data_array import DataArray
from tidy3d.components.geometry.base import Box
from tidy3d.components.grid.grid import Coords
from tidy3d.components.grid.yee_areas import (
    colocated_widths_1d,
    yee_primal_dual_widths_1d,
)
from tidy3d.constants import fp_eps
from tidy3d.exceptions import (
    DataError,
    Tidy3dNotImplementedError,
)

if TYPE_CHECKING:
    from tidy3d.components.data.monitor_data._types import GRID_CORRECTION_TYPE, Coords1D
    from tidy3d.components.data.monitor_data.field.electromagnetic import ElectromagneticFieldData
    from tidy3d.components.grid.grid import Grid
    from tidy3d.components.types import BoundOptional


def _expanded_grid_field_coords(self: ElectromagneticFieldData, field_name: str) -> Coords:
    """Coordinates in the expanded grid corresponding to a given field component."""
    if self.monitor.colocate:
        bounds_dict = self.grid_expanded.boundaries.to_dict
        return Coords(**{key: val[:-1] for key, val in bounds_dict.items()})
    return self.grid_expanded[self.grid_locations[field_name]]


@property
def _grid_correction_dict(self: ElectromagneticFieldData) -> dict[str, GRID_CORRECTION_TYPE]:
    """Return the primal and dual finite grid correction factors as a dictionary."""
    return {
        "grid_primal_correction": self.grid_primal_correction,
        "grid_dual_correction": self.grid_dual_correction,
    }


@property
def _normal_dim(self: ElectromagneticFieldData) -> str:
    """For a 2D monitor data, return the name of the normal dimension. Raise if cannot
    confirm that the associated monitor is 2D."""
    if len(self.monitor.zero_dims) != 1:
        raise DataError("Data must be 2D to get normal dimension.")
    normal_dim = "xyz"[self.monitor.size.index(0)]
    return normal_dim


@property
def _tangential_dims(self: ElectromagneticFieldData) -> list[str]:
    """For a 2D monitor data, return the names of the tangential dimensions. Raise if cannot
    confirm that the associated monitor is 2D."""
    if len(self.monitor.zero_dims) != 1:
        raise DataError("Data must be 2D to get tangential dimensions.")
    tangential_dims = ["x", "y", "z"]
    tangential_dims.pop(self.monitor.zero_dims[0])

    return tangential_dims


def _diff_area_at_yee_positions(
    self: ElectromagneticFieldData, truncate_to_monitor_bounds: bool = False
) -> tuple[DataArray, DataArray, DataArray, DataArray]:
    """Return differential area elements at Yee grid stagger positions.

    The four returned DataArrays correspond to differential areas at the four
    staggered Yee grid locations for field components:
    - dS_EuHv: area elements at Eu/Hv positions = cell_dim1 x dual_dim2
    - dS_EvHu: area elements at Ev/Hu positions = dual_dim1 x cell_dim2
    - dS_Ew: area elements at Ew positions = dual_dim1 x dual_dim2
    - dS_Hw: area elements at Hw positions = cell_dim1 x cell_dim2

    Parameters
    ----------
    truncate_to_monitor_bounds : bool = False
        If True (flux), clamp the integration region to the monitor bounds, further clamped to
        the halo-free colocation grid extent so the integral covers exactly one period
        regardless of monitor size, including ``size=inf``.
        If False (dot/outer_dot), integrate the full grid_expanded extent enclosing the field
        data.

    Returns
    -------
    tuple[DataArray, DataArray, DataArray, DataArray]
        A tuple of (dS_EuHv, dS_EvHu, dS_Ew, dS_Hw) DataArrays with differential
        area elements for Yee grid integration. Each DataArray has dimensions
        corresponding to the two tangential dimensions of the monitor plane.
    """
    if not self.grid_expanded:
        raise DataError(
            "Monitor data requires 'grid_expanded' to compute Yee grid integration sizes."
        )

    _, plane_inds = self.monitor.pop_axis([0, 1, 2], self.monitor.size.index(0.0))
    dims = ["x", "y", "z"]
    mnt_bounds = np.array(self.monitor.bounds)

    cell_sizes = {}
    dual_sizes = {}
    # Field data lives at the grid_expanded positions, which extend one interpolation/halo
    # cell beyond the simulation grid. The per-axis widths come from the shared Yee-width
    # helper; the 1D-degenerate (single-cell) collapse to unit width is handled inside it.
    full_bounds = self.grid_expanded.boundaries.to_dict

    for axis in plane_inds:
        dim = dims[axis]
        if truncate_to_monitor_bounds:
            # Flux: clamp the monitor bounds to the colocation (halo-free) grid extent -- the
            # same extent the colocated ``_diff_area`` integrates -- so the flux covers
            # exactly one period and stays size-invariant, matching the colocated result by
            # construction. All truncate=True callers operate on staggered field data
            # (``monitor.colocate`` is ``False``), so ``colocation_boundaries`` drops the
            # halo symmetrically; revisit this clamp if a ``colocate=True`` caller appears.
            domain_bounds = self.colocation_boundaries.to_dict[dim]
            bounds_kwargs = {
                "mnt_min": mnt_bounds[0, axis],
                "mnt_max": mnt_bounds[1, axis],
                "valid_bounds": (float(domain_bounds[0]), float(domain_bounds[-1])),
            }
        else:
            # dot/outer_dot: integrate the full grid_expanded extent enclosing the data.
            bounds_kwargs = {}
        # A downsampled recording keeps only the monitor's interval_space stride of the grid;
        # the helper then integrates the kept native positions (identity for stride 1).
        num_cells = len(full_bounds[dim]) - 1
        keep_inds = self.monitor.downsample(np.arange(num_cells), axis=axis)
        cell_sizes[dim], dual_sizes[dim] = yee_primal_dual_widths_1d(
            full_bounds[dim], keep_inds=keep_inds, **bounds_kwargs
        )

    dim1 = self._tangential_dims[0]
    dim2 = self._tangential_dims[1]
    dS_EuHv = np.outer(cell_sizes[dim1], dual_sizes[dim2])
    dS_EvHu = np.outer(dual_sizes[dim1], cell_sizes[dim2])
    dS_Ew = np.outer(dual_sizes[dim1], dual_sizes[dim2])
    dS_Hw = np.outer(cell_sizes[dim1], cell_sizes[dim2])

    return (
        DataArray(dS_EuHv, dims=self._tangential_dims),
        DataArray(dS_EvHu, dims=self._tangential_dims),
        DataArray(dS_Ew, dims=self._tangential_dims),
        DataArray(dS_Hw, dims=self._tangential_dims),
    )


@staticmethod
def _clamp_grid_expanded_bounds(
    bounds: BoundOptional,
    grid_expanded: Grid,
    normal_axis: int,
    colocate: bool,
) -> BoundOptional:
    """Clamp solver field bounds so they match the underlying simulation grid.

    ``grid_expanded`` is produced by ``discretize_monitor``, which extends
    the simulation grid by one or more cells for interpolation
    (``_discretize_inds_monitor``).  When a monitor is larger than the
    simulation domain, those extra cells extend past the simulation
    boundaries.

    This method detects bounds that landed on those outermost padding
    boundaries and pulls them inward to the next grid boundary, recovering
    the simulation-grid edges.  The right side is always extended by +1
    cell; the left side is extended by -1 only when ``colocate`` is False.
    """
    rmin, rmax = list(bounds[0]), list(bounds[1])
    _, tangential_axes = Box.pop_axis([0, 1, 2], normal_axis)
    grid_bounds = grid_expanded.boundaries.to_list
    for ax in tangential_axes:
        if rmax[ax] is not None and isclose(
            rmax[ax], grid_bounds[ax][-1], rel_tol=fp_eps, abs_tol=fp_eps
        ):
            rmax[ax] = grid_bounds[ax][-2]
        if (
            not colocate
            and rmin[ax] is not None
            and isclose(rmin[ax], grid_bounds[ax][0], rel_tol=fp_eps, abs_tol=fp_eps)
        ):
            rmin[ax] = grid_bounds[ax][1]
    return (tuple(rmin), tuple(rmax))


@property
def _plane_grid_boundaries(self: ElectromagneticFieldData) -> tuple[Coords1D, Coords1D]:
    """For a 2D monitor data, return the boundaries of the in-plane grid to be used to compute
    differential area and to colocate fields if needed."""
    if np.any(np.array(self.monitor.interval_space) > 1):
        raise Tidy3dNotImplementedError(
            "Cannot determine grid boundaries corresponding to "
            "down-sampled monitor data ('interval_space' > 1 along a direction)."
        )
    dim1, dim2 = self._tangential_dims
    bounds_dict = self.colocation_boundaries.to_dict
    return (bounds_dict[dim1], bounds_dict[dim2])


@property
def _plane_grid_centers(self: ElectromagneticFieldData) -> tuple[Coords1D, Coords1D]:
    """For 2D monitor data, return the centers of the in-plane grid"""
    return [(bs[1:] + bs[:-1]) / 2 for bs in self._plane_grid_boundaries]


@property
def _diff_area(self: ElectromagneticFieldData) -> DataArray:
    """For a 2D monitor data, return the area of each cell in the plane, for use in numerical
    integrations. This assumes that data is colocated to grid boundaries, and uses the
    difference in the surrounding grid centers to compute the area.

    Truncating the cells to the monitor bounds implicitly makes extra pixels which may be
    present have size 0, so they are not included in the integration; for pixels intersected
    by the monitor edge, the size is truncated to the part covered by the monitor. Together
    with integrand values defined at cell boundaries, this realizes the trapezoidal rule with
    the first and last values interpolated to the exact monitor start/end location, provided
    the integrand is zero outside of the monitor geometry -- usually the case for flux and
    dot computations.
    """
    bounds = self._plane_grid_boundaries
    _, plane_inds = self.monitor.pop_axis([0, 1, 2], self.monitor.size.index(0.0))
    mnt_bounds = np.array(self.monitor.bounds)
    mnt_bounds = mnt_bounds[:, plane_inds].T

    sizes_dim0 = colocated_widths_1d(bounds[0], mnt_bounds[0, 0], mnt_bounds[0, 1])
    sizes_dim1 = colocated_widths_1d(bounds[1], mnt_bounds[1, 0], mnt_bounds[1, 1])
    return DataArray(np.outer(sizes_dim0, sizes_dim1), dims=self._tangential_dims)


def _tangential_corrected(
    self: ElectromagneticFieldData, fields: dict[str, DataArray]
) -> dict[str, DataArray]:
    """For a 2D monitor data, extract the tangential components from fields and orient them
    such that the third component would be the normal axis. This just means that the H field
    gets an extra minus sign if the normal axis is ``"y"``. Raise if any of the tangential
    field components is missing.

    The finite grid correction is also applied, so the intended use of these fields is in
    poynting, flux, and dot-like methods. The normal coordinate is dropped from the field data.
    """

    if len(self.monitor.zero_dims) != 1:
        raise DataError("Data must be 2D to get tangential fields.")

    # Tangential field components
    tan_dims = self._tangential_dims
    components = [fname + dim for fname in "EH" for dim in tan_dims]

    normal_dim = self._normal_dim
    normal_axis = "xyz".index(normal_dim)

    tan_fields = {}
    for component in components:
        if component not in fields:
            raise DataError(f"Tangential field component '{component}' missing in field data.")

        correction = 1

        # sign correction to H
        if normal_dim == "y" and component[0] == "H":
            correction *= -1

        # finite grid correction to all fields
        eig_val = self.symmetry_eigenvalues[component](normal_axis)
        if eig_val < 0:
            correction *= self.grid_dual_correction
        else:
            correction *= self.grid_primal_correction

        field_squeezed = fields[component].squeeze(dim=normal_dim, drop=True)
        # TODO DataArray broadcasting here is a slow portion of the dot method
        tan_fields[component] = field_squeezed * correction

    return tan_fields


@property
def _tangential_fields(self: ElectromagneticFieldData) -> dict[str, DataArray]:
    """For a 2D monitor data, get the tangential E and H fields in the 2D plane grid.  Fields
    are oriented such that the third component would be the normal axis. This just means that
    the H field gets an extra minus sign if the normal axis is ``"y"``.

    Note
    ----
        The finite grid correction factors are applied and symmetry is expanded.
    """
    return self._tangential_corrected(self.symmetry_expanded.field_components)


@property
def _colocated_fields(self: ElectromagneticFieldData) -> dict[str, DataArray]:
    """For a 2D monitor data, get all E and H fields colocated to the cell boundaries in the 2D
    plane grid, with symmetries expanded.

    When ``solver_field_bounds`` is set and the monitor was not already colocated,
    clips fields to the valid solver-grid region before interpolation so that
    zero-padded values outside the solver grid do not contaminate the result.
    """

    sym_expanded = self.symmetry_expanded
    field_components = sym_expanded.field_components

    if self.monitor.colocate:
        return field_components

    # Interpolate field components to cell boundaries
    interp_dict = {}
    for dim, bounds in zip(self._tangential_dims, self._plane_grid_boundaries):
        if bounds.size > 1:
            interp_dict[dim] = bounds

    clip_bounds = sym_expanded.solver_field_bounds
    if clip_bounds is not None:
        colocated_fields = {}
        for key, val in field_components.items():
            colocated_fields[key] = val.interp_within_domain(
                interp_dict, clip_bounds, assume_sorted=True
            )
    else:
        interp_dict["assume_sorted"] = True
        colocated_fields = {key: val.interp(**interp_dict) for key, val in field_components.items()}
    return colocated_fields


@property
def _colocated_tangential_fields(self: ElectromagneticFieldData) -> dict[str, DataArray]:
    """For a 2D monitor data, get the tangential E and H fields colocated to the cell boundaries
    in the 2D plane grid.  Fields are oriented such that the third component would be the normal
    axis. This just means that the H field gets an extra minus sign if the normal axis is
    ``"y"``. Raise if any of the tangential field components is missing.

    Note
    ----
        The finite grid correction factors are applied and symmetry is expanded.
    """
    return self._tangential_corrected(self._colocated_fields)


@property
def grid_corrected_copy(self: ElectromagneticFieldData) -> ElectromagneticFieldData:
    """Return a copy of self with grid correction factors applied (if necessary) and symmetry
    expanded."""
    field_data = self.symmetry_expanded_copy
    if len(self.monitor.zero_dims) != 1:
        return field_data

    normal_dim = self._normal_dim
    normal_axis = "xyz".index(normal_dim)
    update = {"grid_primal_correction": 1.0, "grid_dual_correction": 1.0}
    for field_name, field in field_data.field_components.items():
        eig_val = self.symmetry_eigenvalues[field_name](normal_axis)
        if eig_val < 0:
            update[field_name] = field * self.grid_dual_correction
        else:
            update[field_name] = field * self.grid_primal_correction
    return field_data.copy(deep=False, update=update)

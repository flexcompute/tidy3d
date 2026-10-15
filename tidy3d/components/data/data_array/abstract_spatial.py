"""Base class for data arrays on structured spatial grids."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING

import numpy as np

from tidy3d.components.geometry.bound_ops import bounds_contains
from tidy3d.exceptions import DataError
from tidy3d.log import log

from . import _spatial_sampling
from ._plot import _StructuredSpatialPlotProxy
from ._spatial_sampling import _clamp_steps_to_budget, _grid_from_bounds, _grid_steps
from .base import DataArray

if TYPE_CHECKING:
    from xarray.core.types import InterpOptions, Self

    from tidy3d.components.grid.grid import Coords
    from tidy3d.components.types import ArrayLike, Axis, Bound, BoundOptional
    from tidy3d.components.types.base import Coordinate


class AbstractSpatialDataArray(DataArray, ABC):
    """Spatial distribution."""

    __slots__ = ()
    _dims = ("x", "y", "z")
    _data_attrs = {"long_name": "field value"}

    @property
    def plot(self) -> _StructuredSpatialPlotProxy:
        """Plot the spatial data.

        Accepts the same arguments as xarray's ``DataArray.plot()``. The extra
        ``grid`` and ``field`` keyword arguments are accepted for API
        compatibility with :meth:`TriangularGridDataset.plot`, but grid overlay
        is not supported on structured data.

        Parameters
        ----------
        field : bool = True
            Whether to plot the data field. Must be ``True`` for structured
            data.
        grid : bool = False
            Not supported for structured data. Raises ``DataError`` if
            ``True``.
        """
        return _StructuredSpatialPlotProxy(self)

    @property
    def _spatially_sorted(self) -> Self:
        """Check whether sorted and sort if not."""
        needs_sorting = []
        for axis in "xyz":
            axis_coords = np.atleast_1d(self.coords[axis].values)
            if axis_coords.size > 1 and np.any(axis_coords[1:] < axis_coords[:-1]):
                needs_sorting.append(axis)

        if len(needs_sorting) > 0:
            return self.sortby(needs_sorting)

        return self

    def shifted_spatial_coords(self, center: Coordinate) -> Self:
        """Return a copy with spatial coordinates shifted by ``center``."""
        shifted = self
        for axis, dim in enumerate("xyz"):
            if dim in shifted.coords:
                coord_vals = np.asarray(shifted.coords[dim].data)
                shifted = shifted.assign_coords({dim: coord_vals + center[axis]})
        return shifted

    def interpolate_to_grid(
        self,
        grid: Coords,
        *,
        offset: Coordinate | None = None,
        method: InterpOptions = "linear",
        target_dims: tuple[str, ...] | None = None,
    ) -> Self:
        """Interpolate onto a target grid, with optional spatial offset and output ordering."""
        if offset is None:
            interpolated = grid.spatial_interp(self, method)
        else:
            interpolated = grid.spatial_interp(self.shifted_spatial_coords(offset), method)
        if target_dims is not None and tuple(interpolated.dims) != tuple(target_dims):
            interpolated = interpolated.transpose(*target_dims)
        return interpolated

    def interp_within_domain(
        self,
        interp_coords: dict,
        clip_bounds: BoundOptional,
        assume_sorted: bool = False,
    ) -> Self:
        """Interpolate to ``interp_coords``, clipping source data to ``clip_bounds``.

        This is a bounds-aware variant of :meth:`interp` that:

        1. **Clips** the source data to ``clip_bounds``, keeping only data
           points whose coordinates lie on or within the bounds.
        2. **Clamps** all target coordinates to the clipped data range,
           so any target outside that range receives the nearest edge value.
        3. **Interpolates** on the clipped data using the clamped
           coordinates.
        4. **Zeros** the result at target coordinates that are strictly
           outside ``clip_bounds``.

        These steps allow for regular interpolation for strictly interior points,
        while points on the ``clip_bounds`` effectively choose the nearest value from the
        source data, and points strictly outside the bounds are set to 0.

        Each of the six bounds (min/max for x/y/z) is independently
        optional.  A ``None`` value leaves that side unclipped.

        Parameters
        ----------
        interp_coords : dict
            Target interpolation coordinates, keyed by dimension name
            (``"x"``, ``"y"``, ``"z"``).
        clip_bounds : BoundOptional
            ``(min_tuple, max_tuple)`` where each element is a 3-tuple of
            ``Optional[float]``.  Axes set to ``None`` are not clipped.
        assume_sorted : bool = False
            If True, skip sorting of coordinates.

        Returns
        -------
        Self
            Interpolated data with coordinates matching ``interp_coords``.
            Values outside ``clip_bounds`` are zero.
        """
        bound_min, bound_max = clip_bounds

        # Build clip selector from bounds
        clip_sel = {}
        for dim in interp_coords:
            ax = "xyz".index(dim)
            lo = bound_min[ax]
            hi = bound_max[ax]
            if lo is not None or hi is not None:
                clip_sel[dim] = slice(lo, hi)

        clipped = self.sel(clip_sel) if clip_sel else self

        # Clamp target coordinates to clipped data range
        # Emulates picking nearest source data for edge values
        clamped = {
            dim: np.clip(
                vals,
                float(clipped.coords[dim][0]),
                float(clipped.coords[dim][-1]),
            )
            for dim, vals in interp_coords.items()
        }

        result = clipped.interp(**clamped, assume_sorted=assume_sorted)
        # Assign the result coordinates back to the original interpolation points.
        result = result.assign_coords(interp_coords)

        # Zero out values at coordinates strictly outside the clip bounds.
        # Clamping mapped these to edge values, but they should be zero.
        for dim, vals in interp_coords.items():
            ax = "xyz".index(dim)
            lo = bound_min[ax]
            hi = bound_max[ax]
            if lo is None and hi is None:
                continue
            mask = np.zeros(len(vals), dtype=bool)
            if lo is not None:
                mask |= np.asarray(vals) < lo
            if hi is not None:
                mask |= np.asarray(vals) > hi
            if np.any(mask):
                axis_num = result.get_axis_num(dim)
                shape = [1] * result.ndim
                shape[axis_num] = -1
                result.values = np.where(mask.reshape(shape), 0, result.values)

        return result

    def sel_inside(self, bounds: Bound, *, include_interp_padding: bool = True) -> Self:
        """Return a new SpatialDataArray that contains the minimal amount data necessary to cover
        a spatial region defined by ``bounds``. Note that the returned data is sorted with respect
        to spatial coordinates.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        SpatialDataArray
            Extracted spatial data array.
        include_interp_padding : bool = True
            If ``True`` (default), include neighbor points around bounds to support interpolation.
            If ``False``, keep only points whose coordinates are inside bounds.
        """
        if any(bmin > bmax for bmin, bmax in zip(*bounds)):
            raise DataError(
                "Min and max bounds must be packaged as '(minx, miny, minz), (maxx, maxy, maxz)'."
            )

        # make sure data is sorted with respect to coordinates
        sorted_self = self._spatially_sorted

        if not include_interp_padding:
            selected = sorted_self
            for coord, smin, smax, dim in zip(
                (sorted_self.x, sorted_self.y, sorted_self.z),
                bounds[0],
                bounds[1],
                "xyz",
            ):
                coord_vals = np.atleast_1d(coord.values)
                if coord_vals.size <= 1:
                    continue
                selected = selected.sel({dim: slice(smin, smax)})
            return selected

        inds_list = []

        coords = (sorted_self.x, sorted_self.y, sorted_self.z)

        for coord, smin, smax in zip(coords, bounds[0], bounds[1]):
            length = len(coord)

            # one point along direction, assume invariance
            if length == 1:
                comp_inds = [0]
            else:
                # if data does not cover structure at all take the closest index
                if smax < coord[0]:  # structure is completely on the left side
                    # take 2 if possible, so that linear iterpolation is possible
                    comp_inds = np.arange(0, min(2, length))

                elif smin > coord[-1]:  # structure is completely on the right side
                    # take 2 if possible, so that linear iterpolation is possible
                    comp_inds = np.arange(max(0, length - 2), length)

                else:
                    if smin < coord[0]:
                        ind_min = 0
                    else:
                        ind_min = max(0, (coord >= smin).data.argmax() - 1)

                    if smax > coord[-1]:
                        ind_max = length - 1
                    else:
                        ind_max = (coord >= smax).data.argmax()

                    comp_inds = np.arange(ind_min, ind_max + 1)

            inds_list.append(comp_inds)

        return sorted_self.isel(x=inds_list[0], y=inds_list[1], z=inds_list[2])

    def does_cover(self, bounds: Bound, rtol: float = 0.0, atol: float = 0.0) -> bool:
        """Check whether data fully covers specified by ``bounds`` spatial region. If data contains
        only one point along a given direction, then it is assumed the data is constant along that
        direction and coverage is not checked.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.
        rtol : float = 0.0
            Relative tolerance for comparing bounds
        atol : float = 0.0
            Absolute tolerance for comparing bounds

        Returns
        -------
        bool
            Full cover check outcome.
        """
        if any(bmin > bmax for bmin, bmax in zip(*bounds)):
            raise DataError(
                "Min and max bounds must be packaged as '(minx, miny, minz), (maxx, maxy, maxz)'."
            )
        xyz = [self.x, self.y, self.z]
        self_min = [0] * 3
        self_max = [0] * 3
        for dim in range(3):
            coords = xyz[dim]
            if len(coords) == 1:
                self_min[dim] = bounds[0][dim]
                self_max[dim] = bounds[1][dim]
            else:
                self_min[dim] = np.min(coords)
                self_max[dim] = np.max(coords)
        self_bounds = (tuple(self_min), tuple(self_max))
        return bounds_contains(self_bounds, bounds, rtol=rtol, atol=atol)

    def downsample(self, dl: float | ArrayLike, method: InterpOptions = "linear") -> Self:
        """Resample onto a uniform Cartesian grid spanning the same bounds, at spacing ``dl``.

        Useful for shrinking a finely resolved field before it is embedded in a simulation --
        e.g. a charge density handed to
        :meth:`Simulation.perturbed_mediums_copy <tidy3d.Simulation.perturbed_mediums_copy>`,
        where the full-resolution field can dominate the uploaded file size. Sampling is at the
        grid nodes, not averaged over cells, so it does not conserve the integral of the field:
        a feature thinner than ``dl`` is erased or, if a node lands on it, widened to ``dl``.
        Check the result before relying on it.

        The grid spans the original bounds exactly, so the spacing it lands on is ``dl`` rounded
        down to the nearest whole number of steps across each axis, never up. Axes are only ever
        made coarser: one whose own spacing is already at or above ``dl`` keeps its coordinate,
        and only a degenerate one reduces to a single point. An axis is compared against its
        finest gap, so a nonuniform one is resampled over its whole span and can come back with
        more points than it started with; a ``dl`` that would exceed the largest array that can
        be sampled -- or this field's own size, if it is already larger -- is raised until it
        fits, with a warning. Non-spatial dimensions (``f``, ``t``, ...) pass through untouched.

        Parameters
        ----------
        dl : Union[float, ArrayLike]
            Target grid spacing (micron), either isotropic or per ``x``, ``y``, ``z``.
        method : InterpOptions = "linear"
            Interpolation method used to sample onto the new grid.

        Returns
        -------
        AbstractSpatialDataArray
            Data sampled on the coarsened grid.

        Example
        -------
        >>> from tidy3d import SpatialDataArray
        >>> x = np.linspace(0, 1, 101)
        >>> coords = dict(x=x, y=x, z=[0.0])
        >>> data = SpatialDataArray(np.random.random((101, 101, 1)), coords=coords)
        >>> coarse = data.downsample(dl=0.1)
        >>> coarse.shape
        (11, 11, 1)
        """
        sorted_self = self._spatially_sorted
        current = {dim: np.atleast_1d(sorted_self.coords[dim].values) for dim in "xyz"}
        bounds = (
            tuple(float(current[dim][0]) for dim in "xyz"),
            tuple(float(current[dim][-1]) for dim in "xyz"),
        )
        steps = _grid_steps(dl)

        # Compared against the finest gap, not the point count: an axis can hold few points and
        # still resolve something the request would erase, as [0, 0.01, 1] does at 'dl=0.4'.
        keep_coords = {
            dim: current[dim]
            for ind, dim in enumerate("xyz")
            if len(current[dim]) < 2 or np.min(np.diff(current[dim])) >= steps[ind]
        }
        if len(keep_coords) == 3:
            return sorted_self

        # A nonuniform axis is resampled over its whole span whenever its finest gap is below
        # 'dl', so the grid can run past the cap; clamp it here rather than let
        # '_grid_from_bounds' reject it, naming arguments this method does not take.
        resampled = np.array([dim not in keep_coords for dim in "xyz"])
        extents = np.array([current[dim][-1] - current[dim][0] for dim in "xyz"], dtype=float)
        # Kept axes are not coarsened, so a field already above the cap could never be sampled
        # at all; its own size is the ceiling instead, which is always reachable because every
        # resampled axis can fall back to its two endpoints.
        budget = max(
            _spatial_sampling.MAX_SPATIAL_SAMPLES,
            int(np.prod([len(current[dim]) for dim in "xyz"])),
        )
        raised = _clamp_steps_to_budget(
            steps[resampled],
            extents[resampled],
            budget=budget,
            fixed_total=np.prod([len(coord) for coord in keep_coords.values()]),
        )
        if raised is not None:
            steps = np.array(steps, dtype=float)
            steps[resampled], _ = raised
            log.warning(
                f"Sampling at '{dl}' needs more than the {budget} points supported in one "
                f"array; it has been clamped to {np.array2string(steps, precision=4)}. "
                "Specify a larger down-sampling step to suppress this warning.",
                log_once=True,
            )

        coords = dict(
            zip(
                "xyz",
                _grid_from_bounds(
                    bounds=bounds,
                    resolution=steps,
                    keep_coords=keep_coords,
                    max_samples=budget,
                ),
            )
        )

        # leave untouched axes out of the interpolation: a degenerate one has nothing to
        # interpolate along and makes scipy divide by a zero-width interval
        changed = {dim: values for dim, values in coords.items() if dim not in keep_coords}
        return sorted_self.interp(**changed, method=method)

    def reflect(
        self, axis: Axis, center: float, reflection_only: bool = False, symmetry: float = 1
    ) -> Self:
        """Reflect data across the plane define by parameters ``axis`` and ``center`` from right to
        left. Note that the returned data is sorted with respect to spatial coordinates.

        Parameters
        ----------
        axis : Literal[0, 1, 2]
            Normal direction of the reflection plane.
        center : float
            Location of the reflection plane along its normal direction.
        reflection_only : bool = False
            Return only reflected data.
        symmetry : float = 1
            Symmetry factor of the reflection.

        Returns
        -------
        AbstractSpatialDataArray
            Data after reflection is performed.
        """

        sorted_self = self._spatially_sorted
        dim = "xyz"[axis]

        # index the reflected axis by name so extra dims ('voltage', 't', ...) pass through
        coords = {name: np.asarray(sorted_self.coords[name].data) for name in sorted_self.dims}
        data = np.array(sorted_self.data)
        dim_index = list(sorted_self.dims).index(dim)

        data_left_bound = coords[dim][0]

        if np.isclose(center, data_left_bound):
            num_duplicates = 1
        elif center > data_left_bound:
            raise DataError("Reflection center must be outside and to the left of the data region.")
        else:
            num_duplicates = 0

        if reflection_only:
            new_coords = dict(coords)
            new_coords[dim] = 2 * center - coords[dim]
            return type(self)(data * symmetry, coords=new_coords).sortby(dim)

        shape = list(np.shape(data))
        old_len = shape[dim_index]
        shape[dim_index] = 2 * old_len - num_duplicates

        ind_left = [slice(None)] * data.ndim
        ind_right = [slice(None)] * data.ndim

        ind_left[dim_index] = slice(old_len - 1, None, -1)
        ind_right[dim_index] = slice(old_len - num_duplicates, None)

        # source dtype: a default-float zeros() would discard the imaginary part
        new_data = np.zeros(shape, dtype=data.dtype)

        new_data[tuple(ind_left)] = data * symmetry
        new_data[tuple(ind_right)] = data

        new_dim_coords = np.zeros(shape[dim_index])
        new_dim_coords[old_len - num_duplicates :] = coords[dim]
        new_dim_coords[old_len - 1 :: -1] = 2 * center - coords[dim]

        new_coords = dict(coords)
        new_coords[dim] = new_dim_coords

        return type(self)(new_data, coords=new_coords)

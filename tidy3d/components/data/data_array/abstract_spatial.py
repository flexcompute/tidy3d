"""Base class for data arrays on structured spatial grids."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from flexcompute.data.arrays.abstract_spatial import (
    AbstractSpatialDataArray as CoreAbstractSpatialDataArray,
)
from flexcompute.data.arrays.abstract_spatial import (
    SpatialInterpolator,
)

from tidy3d.log import log

from . import _spatial_sampling
from ._plot import _StructuredSpatialPlotProxy
from ._spatial_sampling import _clamp_steps_to_budget, _grid_from_bounds, _grid_steps
from .base import DataArray

__all__ = ["AbstractSpatialDataArray", "SpatialInterpolator"]

if TYPE_CHECKING:
    from xarray.core.types import InterpOptions, Self

    from tidy3d.components.types import ArrayLike


class AbstractSpatialDataArray(DataArray, CoreAbstractSpatialDataArray):
    """Spatial distribution."""

    __slots__ = ()

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

"""Monitor level data, store the DataArrays associated with a single heat-charge monitor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field, model_validator

from tidy3d.components.base_sim.data.monitor_data import AbstractUnstructuredMonitorData
from tidy3d.components.data.data_array import SpatialDataArray
from tidy3d.components.data.utils import TetrahedralGridDataset, TriangularGridDataset
from tidy3d.components.tcad.types import HeatChargeMonitorType
from tidy3d.components.types import TYPE_TAG_STR, ScalarSymmetry
from tidy3d.components.types.base import discriminated_union
from tidy3d.exceptions import DataError
from tidy3d.log import log

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Literal

    from tidy3d.compat import Self
    from tidy3d.components.types import ArrayLike, Bound

FieldDataset = SpatialDataArray | discriminated_union(
    TriangularGridDataset | TetrahedralGridDataset
)
UnstructuredFieldType = TriangularGridDataset | TetrahedralGridDataset

# Ceiling on the Cartesian grid ``to_spatial_data_array`` will build. The array is held in
# memory whole and interpolated point by point, and a bounds/resolution pair is taken
# literally, so an innocent-looking resolution over a device-sized box can ask for
# terabytes.
MAX_SPATIAL_SAMPLES = 4_000_000

# Tolerance for matching a requested bias against the recorded ones. Both travel as IEEE
# doubles, so this only absorbs float noise -- it is not a "nearest bias" search.
VOLTAGE_MATCH_TOL = 1e-9


class HeatChargeMonitorData(AbstractUnstructuredMonitorData, ABC):
    """Abstract base class of objects that store data pertaining to a single :class:`HeatChargeMonitor`."""

    monitor: HeatChargeMonitorType = Field(
        discriminator=TYPE_TAG_STR,
        title="Monitor",
        description="Monitor associated with the data.",
    )

    symmetry: tuple[ScalarSymmetry, ScalarSymmetry, ScalarSymmetry] = Field(
        (0, 0, 0),
        title="Symmetry",
        description="Symmetry of the original simulation in x, y, and z.",
    )

    @abstractmethod
    def field_components(self) -> dict:
        """Maps the field components to their associated data."""

    @model_validator(mode="after")
    def _warn_missing_fields(self) -> Self:
        """Warn when a monitor field has no data available."""
        for field_name, field_data in self.field_components.items():
            if field_data is None:
                log.warning(
                    f"No data is available for monitor '{self.monitor.name}' field '{field_name}'. "
                    "This is typically caused by monitor not intersecting any solid medium."
                )
        return self

    def field_name(self, val: str = "") -> str:
        """Gets the name of the fields to be plot."""
        fields = self.field_components.keys()
        name = ""
        for field in fields:
            if val == "abs^2":
                name = f"{field}²"
            else:
                name = f"{field}"
        return name

    @property
    def symmetry_expanded_copy(self) -> HeatChargeMonitorData:
        """Return copy of self with symmetry applied."""

        new_field_components = {}
        for field, val in self.field_components.items():
            # Unpopulated fields stay unpopulated in the expanded copy.
            new_field_components[field] = (
                self._symmetry_expanded_copy_base(data=val) if val is not None else None
            )

        return self.updated_copy(
            symmetry=(0, 0, 0), **new_field_components, deep=False, validate=False
        )

    def to_spatial_data_array(
        self,
        x: ArrayLike | None = None,
        y: ArrayLike | None = None,
        z: ArrayLike | None = None,
        *,
        field: str | None = None,
        bounds: Bound | None = None,
        resolution: float | tuple[float, float, float] | None = None,
        voltage: float | None = None,
        fill_value: float = 0.0,
        method: Literal["linear"] = "linear",
    ) -> SpatialDataArray:
        """Sample a recorded field onto a Cartesian grid.

        Solver output lives on an unstructured grid, while the objects that consume a field
        as an input -- :class:`HeatSource`, :class:`CustomDoping` -- take a Cartesian
        :class:`SpatialDataArray`. This resamples between the two. Specify the target grid
        either as explicit coordinate arrays (``x``, ``y``, ``z``) or as ``bounds`` plus
        ``resolution``.

        Parameters
        ----------
        x : ArrayLike = None
            x-coordinates of the target grid. Provide either all or none of ``x``, ``y``, ``z``.
        y : ArrayLike = None
            y-coordinates of the target grid.
        z : ArrayLike = None
            z-coordinates of the target grid.
        field : str = None
            Which recorded field to convert, by its name in ``field_components``. Optional
            when the data holds exactly one field.
        bounds : Bound = None
            Target grid extent as ``((xmin, ymin, zmin), (xmax, ymax, zmax))``. Requires
            ``resolution`` and is mutually exclusive with ``x``/``y``/``z``.
        resolution : Union[float, tuple[float, float, float]] = None
            Target grid step, either isotropic or per axis. Used with ``bounds``.
        voltage : float = None
            Bias point to extract. Required when the data holds more than one bias.
        fill_value : float = 0.0
            Value assigned to grid points outside the solver's grid. Zero suits a source
            term, where no solution means no contribution; extrapolating instead would
            extend the field beyond the region it was solved on.
        method : Literal["linear"] = "linear"
            Interpolation method. Only ``"linear"`` is supported: nearest-neighbour
            interpolation cannot honour ``fill_value``.

        Returns
        -------
        :class:`.SpatialDataArray`
            The field on the requested Cartesian grid.

        Notes
        -----
            Feeding the result into another simulation interpolates a second time, onto that
            simulation's mesh. Detail lost here cannot be recovered downstream, so choose a
            resolution fine enough to resolve the structure of the field.
        """
        if method != "linear":
            raise DataError(
                f"'method' must be 'linear' (got '{method}'). Nearest-neighbour "
                "interpolation ignores 'fill_value' and would extend the field beyond the "
                "region it was solved on."
            )

        data = self._resolve_field(field=field)

        coords_given = any(comp is not None for comp in (x, y, z))
        if coords_given and any(comp is None for comp in (x, y, z)):
            raise DataError("Must provide either all or none of 'x', 'y', and 'z'.")
        if coords_given and (bounds is not None or resolution is not None):
            raise DataError(
                "Provide the target grid either as 'x'/'y'/'z' or as 'bounds' and "
                "'resolution', not both."
            )
        if not coords_given and (bounds is None) != (resolution is None):
            raise DataError("'bounds' and 'resolution' must be provided together.")

        if not coords_given and bounds is not None:
            x, y, z = self._grid_from_bounds(bounds=bounds, resolution=resolution)
            coords_given = True

        if coords_given:
            # A scalar coordinate makes xarray drop that dimension, which the Cartesian
            # branch below cannot transpose back. The unstructured path normalizes
            # internally; do it here so both accept the scalars the signature advertises.
            x, y, z = (np.atleast_1d(comp) for comp in (x, y, z))
            self._check_sample_count([x, y, z])

        if not isinstance(data, (TriangularGridDataset, TetrahedralGridDataset)):
            # Already Cartesian. Only a plain 'SpatialDataArray' can be returned as one --
            # anything carrying extra dimensions (a time series, say) would need a selector
            # for them, which this converter does not take.
            if not isinstance(data, SpatialDataArray):
                extra = [dim for dim in data.dims if dim not in ("x", "y", "z")]
                raise DataError(
                    f"The data for monitor '{self.monitor.name}' is a "
                    f"'{type(data).__name__}' carrying {extra} beyond the spatial "
                    "dimensions, which has no single 'SpatialDataArray' form. Select along "
                    f"{extra} first."
                )
            # Only resample if a different grid was asked for.
            if not coords_given:
                return data
            return SpatialDataArray(
                data.interp(
                    x=x, y=y, z=z, method=method, kwargs={"fill_value": fill_value}
                ).transpose("x", "y", "z")
            )

        if not coords_given:
            raise DataError(
                "The data is stored on an unstructured grid, so a target Cartesian grid "
                "must be given through 'x'/'y'/'z' or 'bounds' and 'resolution'."
            )

        data = self._select_voltage(data=data, voltage=voltage)
        # The same guard the Cartesian branch applies. Vector data (an electric field, a
        # current density) keeps its 'axis' dimension through '_select_voltage', which only
        # drops 'voltage', so interpolating here would return a bias-selected 4D array from a
        # method that promises a 'SpatialDataArray'.
        extra_dims = data._non_spatial_dims
        if extra_dims:
            raise DataError(
                f"The data for monitor '{self.monitor.name}' is a "
                f"'{type(data).__name__}' carrying {extra_dims} beyond the spatial "
                "dimensions, which has no single 'SpatialDataArray' form. A vector field "
                "carries 'axis' (0, 1, 2 for x, y, z); select one component before "
                "converting."
            )
        return data.interp(x=x, y=y, z=z, fill_value=fill_value, method=method)

    def _resolve_field(self, field: str | None) -> FieldDataset:
        """Pick the field to convert, requiring a name only when the choice is ambiguous."""
        components = self.field_components
        if not components:
            # 'pass field to choose one' cannot be followed when there is nothing to choose
            # from: a capacitance monitor records values against bias, not a spatial field.
            raise DataError(
                f"Monitor '{self.monitor.name}' records no spatial field, so it has no "
                "'SpatialDataArray' form."
            )
        if field is None:
            if len(components) != 1:
                raise DataError(
                    f"Monitor '{self.monitor.name}' holds {len(components)} fields "
                    f"({sorted(components)}); pass 'field' to choose one."
                )
            field = next(iter(components))
        elif field not in components:
            raise DataError(
                f"'{field}' is not a field of monitor '{self.monitor.name}'; "
                f"available: {sorted(components)}."
            )

        data = components[field]
        if data is None:
            raise DataError(
                f"No data is available for monitor '{self.monitor.name}' field '{field}', "
                "so it cannot be converted to a 'SpatialDataArray'."
            )
        return data

    @staticmethod
    def _check_sample_count(coords: Sequence[ArrayLike]) -> None:
        """Reject a target grid too large to materialise."""
        counts = [len(np.atleast_1d(comp)) for comp in coords]
        total = counts[0] * counts[1] * counts[2]
        if total > MAX_SPATIAL_SAMPLES:
            raise DataError(
                f"The requested grid has {counts[0]}x{counts[1]}x{counts[2]} = {total} "
                f"points, above the {MAX_SPATIAL_SAMPLES} supported. Coarsen 'resolution' "
                "or narrow 'bounds'; the array is held in memory whole and sampled point "
                "by point."
            )

    @staticmethod
    def _grid_from_bounds(
        bounds: Bound, resolution: float | tuple[float, float, float]
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Build per-axis coordinate arrays spanning ``bounds`` at ``resolution``."""
        # Checked before unpacking: a malformed pair would otherwise surface as a bare
        # 'not enough values to unpack' from somewhere downstream.
        if len(bounds) != 2 or any(len(corner) != 3 for corner in bounds):
            raise DataError(
                f"'bounds' must be ((xmin, ymin, zmin), (xmax, ymax, zmax)); got {bounds!r}."
            )

        rmin, rmax = bounds
        steps = (resolution,) * 3 if np.isscalar(resolution) else tuple(resolution)
        if len(steps) != 3:
            raise DataError("'resolution' must be a scalar or a 3-tuple.")
        if any(step <= 0 for step in steps):
            raise DataError("'resolution' must be positive.")

        # Sizes first, allocation second: the whole point of the cap is to refuse before
        # anything large is materialised, and np.linspace on one enormous axis would
        # already have exhausted memory by the time a post-hoc check ran.
        counts = []
        for axis, (lo, hi, step) in enumerate(zip(rmin, rmax, steps)):
            extent = hi - lo
            if extent < 0:
                # Kept distinct from an equal pair: swapped bounds would otherwise collapse
                # to a single plane and quietly return a field over the wrong region.
                raise DataError(
                    f"'bounds' is reversed along {'xyz'[axis]}: the minimum ({lo}) is above "
                    f"the maximum ({hi}). Expected ((xmin, ymin, zmin), (xmax, ymax, zmax))."
                )
            # A zero-thickness axis (a 2D simulation) collapses to one sample plane.
            counts.append(1 if extent == 0 else max(2, int(np.ceil(extent / step)) + 1))

        total = counts[0] * counts[1] * counts[2]
        if total > MAX_SPATIAL_SAMPLES:
            raise DataError(
                f"'bounds' at this 'resolution' needs {counts[0]}x{counts[1]}x{counts[2]} = "
                f"{total} points, above the {MAX_SPATIAL_SAMPLES} supported. Coarsen "
                "'resolution' or narrow 'bounds'."
            )

        return tuple(
            np.array([lo]) if count == 1 else np.linspace(lo, hi, count)
            for lo, hi, count in zip(rmin, rmax, counts)
        )

    def _select_voltage(
        self, data: UnstructuredFieldType, voltage: float | None
    ) -> UnstructuredFieldType:
        """Reduce a bias-resolved dataset to the single requested bias point."""
        if "voltage" not in data._non_spatial_dims:
            return data

        voltages = np.atleast_1d(data.values.coords["voltage"].data)
        if voltage is None:
            if len(voltages) > 1:
                raise DataError(
                    f"The data for monitor '{self.monitor.name}' holds {len(voltages)} bias "
                    f"points ({voltages.tolist()}). Pass 'voltage' to select one."
                )
            index = 0
        else:
            # Matched, not snapped: silently returning the nearest bias would hand back a
            # field for a different operating point than the one asked for.
            matches = np.flatnonzero(
                np.isclose(voltages, voltage, rtol=0.0, atol=VOLTAGE_MATCH_TOL)
            )
            if matches.size == 0:
                raise DataError(
                    f"No recorded bias matches voltage={voltage} for monitor "
                    f"'{self.monitor.name}'; available: {voltages.tolist()}."
                )
            index = int(matches[0])

        # ``drop=True`` removes the voltage dimension, so the interpolation downstream
        # returns a plain 'SpatialDataArray' rather than a bias-resolved array.
        return data.isel(voltage=index, drop=True)

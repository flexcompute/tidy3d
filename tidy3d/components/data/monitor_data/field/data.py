from __future__ import annotations

from typing import TYPE_CHECKING, Any

import autograd.numpy as np
from pydantic import Field

from tidy3d.components.autograd.source_factory import current_component_data_array
from tidy3d.components.data.dataset import FieldDataset
from tidy3d.components.data.em_fields import frequency_normalized_field_components
from tidy3d.components.grid.grid import Coords
from tidy3d.components.monitor import FieldMonitor
from tidy3d.components.source.current import CustomCurrentSource
from tidy3d.components.source.field import CustomFieldSource
from tidy3d.components.source.time import GaussianPulse
from tidy3d.components.validators import enforce_monitor_fields_present
from tidy3d.exceptions import AdjointError

from .electromagnetic import ElectromagneticFieldData

if TYPE_CHECKING:
    from collections.abc import Callable

    from tidy3d.components.data.data_array import ScalarFieldDataArray
    from tidy3d.components.source.time import SourceTimeType
    from tidy3d.components.types import Coordinate, Size


class FieldData(FieldDataset, ElectromagneticFieldData):
    """
    Data associated with a :class:`.FieldMonitor`: scalar components of E and H fields.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

        This dataset can contain all electric and magnetic field components: ``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``,
        and ``Hz``.

    Example
    -------
    >>> from tidy3d import Grid, ScalarFieldDataArray
    >>> from tidy3d.components.grid.grid import Coords
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], f=f)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> scalar_field = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> monitor = FieldMonitor(
    ...     size=(2,4,6), freqs=[2e14, 3e14], name='field', fields=['Ex', 'Hz'], colocate=True
    ... )
    >>> data = FieldData(monitor=monitor, Ex=scalar_field, Hz=scalar_field, grid_expanded=grid)

    .. TODO sort out standalone data example.

    See Also
    --------

    **Notebooks:**
        * `Quickstart <../../notebooks/StartHere.html>`_: Usage in a basic simulation flow.
        * `Performing visualization of simulation data <../../notebooks/VizData.html>`_
        * `Advanced monitor data manipulation and visualization <../../notebooks/XarrayTutorial.html>`_
    """

    monitor: FieldMonitor = Field(
        title="Monitor",
        description="Frequency-domain field monitor associated with the data.",
    )

    _contains_monitor_fields = enforce_monitor_fields_present()

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> FieldDataset:
        """Return copy of self after normalization is applied using source spectrum function."""
        return self.copy(
            deep=False,
            update=frequency_normalized_field_components(self.field_components, source_spectrum_fn),
        )

    def to_source(
        self, source_time: SourceTimeType, center: Coordinate, size: Size = None, **kwargs: Any
    ) -> CustomFieldSource:
        """Create a :class:`.CustomFieldSource` from the fields stored in the :class:`.FieldData`.

        Parameters
        ----------
        source_time: :class:`.SourceTime`
            Specification of the source time-dependence.
        center: tuple[float, float, float]
            Source center in x, y and z.
        size: tuple[float, float, float]
            Source size in x, y, and z. If not provided, the size of the monitor associated to the
            data is used.
        **kwargs
            Extra keyword arguments passed to :class:`.CustomFieldSource`.

        Returns
        -------
        :class:`.CustomFieldSource`
            Source injecting the fields stored in the :class:`.FieldData`, with other settings as
            provided in the input arguments.
        """

        if not size:
            size = self.monitor.size

        fields = {}
        for name, field in self.symmetry_expanded.field_components.items():
            fields[name] = field.copy()
            for dim, dim_name in enumerate("xyz"):
                coords_shift = field.coords[dim_name] - self.monitor.center[dim]
                fields[name].coords[dim_name] = coords_shift

        dataset = FieldDataset(**fields)
        return CustomFieldSource(
            field_dataset=dataset, source_time=source_time, center=center, size=size, **kwargs
        )

    def _make_adjoint_sources(
        self,
        dataset_names: list[str],
        fwidth: float,
        simulation_bounds: tuple[Coordinate, Coordinate] | None = None,
    ) -> list[CustomCurrentSource]:
        """Converts a :class:`.FieldData` to a list of adjoint current or point sources."""

        sources = []
        source_geo = self.monitor.geometry
        freqs = self.monitor.freqs

        for freq0 in freqs:
            src_field_components = {}
            for name, field_component in self.field_components.items():
                # get the VJP values at frequency and apply adjoint phase
                field_component = field_component.sel(f=freq0)

                # accounts for the effective size of the source when injecting into a
                # simulation with symmetry
                symmetry_factor = np.prod(field_component.values.shape) / np.prod(
                    self.symmetry_expanded.field_components[name].sel(f=freq0).values.shape
                )
                source_data = current_component_data_array(
                    component=name,
                    base_values=field_component.values,
                    spatial_coords={key: np.array(field_component.coords[key]) for key in "xyz"},
                    source_center=source_geo.center,
                    freq=float(freq0),
                    symmetry_factor=float(symmetry_factor),
                )
                if source_data is not None:
                    src_field_components[name] = source_data

            # dont include this source if no data
            if not src_field_components:
                continue

            component_groups = [src_field_components]
            if not self._adjoint_source_union_is_safe(src_field_components, simulation_bounds):
                component_groups = [
                    {component: source_data}
                    for component, source_data in src_field_components.items()
                ]

            for component_group in component_groups:
                center, size, fitted_components = self._fit_adjoint_source_support(
                    component_group, simulation_bounds
                )
                sources.append(
                    CustomCurrentSource(
                        center=center,
                        size=size,
                        source_time=GaussianPulse(
                            freq0=freq0,
                            fwidth=fwidth,
                        ),
                        current_dataset=FieldDataset(**fitted_components),
                        interpolate=True,
                        confine_to_bounds=True,
                    )
                )

        return sources

    def _adjoint_source_union_is_safe(
        self,
        field_components: dict[str, ScalarFieldDataArray],
        simulation_bounds: tuple[Coordinate, Coordinate] | None = None,
    ) -> bool:
        """Whether one union box adds no Yee samples outside any component's support."""
        if len(field_components) <= 1:
            return True

        source_geo = self.monitor.geometry
        single_sample_bounds = self._adjoint_single_sample_support_bounds(
            field_components=field_components, simulation_bounds=simulation_bounds
        )
        component_sample_bounds = {
            name: {
                dim: (
                    float(field.coords[dim].min()),
                    float(field.coords[dim].max()),
                )
                for dim in "xyz"
            }
            for name, field in field_components.items()
        }
        union_sample_bounds = {
            dim: (
                min(bounds[dim][0] for bounds in component_sample_bounds.values()),
                max(bounds[dim][1] for bounds in component_sample_bounds.values()),
            )
            for dim in "xyz"
        }
        component_support_bounds = {
            name: {
                dim: self._adjoint_component_support_bounds(
                    field=field,
                    dim=dim,
                    single_sample_bounds=single_sample_bounds,
                )
                for axis, dim in enumerate("xyz")
            }
            for name, field in field_components.items()
        }
        union_support_bounds = {
            dim: (
                min(bounds[dim][0] for bounds in component_support_bounds.values()),
                max(bounds[dim][1] for bounds in component_support_bounds.values()),
            )
            for dim in "xyz"
        }

        component_bounds_match_union = all(
            np.allclose(
                component_sample_bounds[name][dim],
                union_sample_bounds[dim],
                rtol=0.0,
                atol=1e-12,
            )
            and np.allclose(
                component_support_bounds[name][dim],
                union_support_bounds[dim],
                rtol=0.0,
                atol=1e-12,
            )
            for name in field_components
            for axis, dim in enumerate("xyz")
            if source_geo.size[axis] != 0
        )

        if self.grid_expanded is None or self.monitor.colocate:
            return component_bounds_match_union

        for name in field_components:
            yee_coords = self.grid_expanded[self.grid_locations[name]].to_dict
            for axis, dim in enumerate("xyz"):
                if source_geo.size[axis] == 0:
                    continue
                lower_union, upper_union = union_support_bounds[dim]
                lower_component, upper_component = component_sample_bounds[name][dim]
                scale = max(
                    1.0,
                    abs(lower_union),
                    abs(upper_union),
                    abs(lower_component),
                    abs(upper_component),
                )
                tolerance = 1e-12 * scale
                coords_local = np.asarray(yee_coords[dim], dtype=float) - source_geo.center[axis]
                in_union = (coords_local >= lower_union - tolerance) & (
                    coords_local <= upper_union + tolerance
                )
                union_yee_coords = coords_local[in_union]
                if np.any(union_yee_coords < lower_component - tolerance) or np.any(
                    union_yee_coords > upper_component + tolerance
                ):
                    return False
        return True

    def _fit_adjoint_source_support(
        self,
        field_components: dict[str, ScalarFieldDataArray],
        simulation_bounds: tuple[Coordinate, Coordinate] | None = None,
    ) -> tuple[Coordinate, Coordinate, dict[str, ScalarFieldDataArray]]:
        """Fit and rebase a current source to the union of component cell supports."""
        source_geo = self.monitor.geometry
        center = list(source_geo.center)
        size = list(source_geo.size)
        midpoints = [0.0, 0.0, 0.0]
        single_sample_bounds = self._adjoint_single_sample_support_bounds(
            field_components=field_components, simulation_bounds=simulation_bounds
        )

        for axis, dim in enumerate("xyz"):
            if source_geo.size[axis] == 0:
                continue
            component_bounds = [
                self._adjoint_component_support_bounds(
                    field=field,
                    dim=dim,
                    single_sample_bounds=single_sample_bounds,
                )
                for field in field_components.values()
            ]
            lower = min(bounds[0] for bounds in component_bounds)
            upper = max(bounds[1] for bounds in component_bounds)
            midpoints[axis] = 0.5 * (lower + upper)
            center[axis] += midpoints[axis]
            size[axis] = upper - lower

        fitted_components = {}
        for name, field in field_components.items():
            new_coords = {}
            for axis, dim in enumerate("xyz"):
                if source_geo.size[axis] != 0:
                    new_coords[dim] = np.asarray(field.coords[dim], dtype=float) - midpoints[axis]
            fitted_field = field.assign_coords(new_coords) if new_coords else field
            fitted_components[name] = fitted_field

        return tuple(center), tuple(size), fitted_components

    def _adjoint_single_sample_support_bounds(
        self,
        field_components: dict[str, ScalarFieldDataArray],
        simulation_bounds: tuple[Coordinate, Coordinate] | None,
    ) -> dict[str, tuple[float, float]]:
        """Resolve physical support bounds for source axes with one sample."""
        source_geo = self.monitor.geometry
        support_bounds = {}
        for axis, dim in enumerate("xyz"):
            has_single_sample = any(
                np.asarray(field.coords[dim]).size == 1 for field in field_components.values()
            )
            if not has_single_sample:
                continue

            if np.isfinite(source_geo.size[axis]):
                lower = float(source_geo.bounds[0][axis] - source_geo.center[axis])
                upper = float(source_geo.bounds[1][axis] - source_geo.center[axis])
                single_coords = []
                for field in field_components.values():
                    coords = np.asarray(field.coords[dim], dtype=float)
                    if coords.size == 1:
                        single_coords.append(float(coords[0]))
                lower = min(lower, *single_coords)
                upper = max(upper, *single_coords)
            else:
                if simulation_bounds is None:
                    raise AdjointError(
                        "Cannot determine adjoint current-source support for a single-sample "
                        f"infinite FieldMonitor axis '{dim}' without simulation bounds. "
                        "Build adjoint sources through SimulationData or pass simulation_bounds."
                    )
                lower = float(simulation_bounds[0][axis] - source_geo.center[axis])
                upper = float(simulation_bounds[1][axis] - source_geo.center[axis])

            support_bounds[dim] = (lower, upper)
        return support_bounds

    @staticmethod
    def _adjoint_component_support_bounds(
        field: ScalarFieldDataArray,
        dim: str,
        single_sample_bounds: dict[str, tuple[float, float]],
    ) -> tuple[float, float]:
        """Return physical support bounds for one component along a source axis."""
        coords = np.asarray(field.coords[dim], dtype=float)
        if coords.size == 1:
            return single_sample_bounds[dim]

        cell_sizes = Coords(
            **{axis_dim: np.asarray(field.coords[axis_dim], dtype=float) for axis_dim in "xyz"}
        ).cell_sizes
        widths = np.asarray(cell_sizes[dim], dtype=float)
        lower_edges = coords - 0.5 * widths
        upper_edges = coords + 0.5 * widths
        return float(np.min(lower_edges)), float(np.max(upper_edges))

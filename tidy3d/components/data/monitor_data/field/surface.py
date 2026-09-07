from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
import xarray as xr
from pydantic import Field

from tidy3d.components.base import TYPE_TAG_STR
from tidy3d.components.base_sim.data.monitor_data import AbstractUnstructuredMonitorData
from tidy3d.components.data.data_array import FreqDataArray
from tidy3d.components.data.dataset import ElectromagneticSurfaceFieldDataset
from tidy3d.components.data.monitor_data.base import MonitorData
from tidy3d.components.monitor import (
    SurfaceFieldMonitor,
    SurfaceFieldTimeMonitor,
)
from tidy3d.components.validators import enforce_monitor_fields_present
from tidy3d.exceptions import DataError

if TYPE_CHECKING:
    from collections.abc import Callable

    from tidy3d.components.data.monitor_data.base import AbstractFieldData
    from tidy3d.components.data.unstructured.surface import TriangularSurfaceDataset


class ElectromagneticSurfaceFieldData(
    MonitorData, AbstractUnstructuredMonitorData, ElectromagneticSurfaceFieldDataset, ABC
):
    """Collection of vector fields on a surface with some symmetry properties."""

    monitor: SurfaceFieldMonitor | SurfaceFieldTimeMonitor = Field(discriminator=TYPE_TAG_STR)

    _contains_monitor_fields = enforce_monitor_fields_present()

    @property
    def symmetry_expanded(self) -> ElectromagneticSurfaceFieldData:
        """Return the :class:`.ElectromagneticSurfaceFieldData` with fields expanded based on symmetry. If
        any symmetry is nonzero (i.e. expanded), the interpolation implicitly creates a copy of the
        data array. However, if symmetry is not expanded, the returned array contains a view of
        the data, not a copy.

        Returns
        -------
        :class:`ElectromagneticSurfaceFieldData`
            A data object with the symmetry expanded fields.
        """

        if all(sym == 0 for sym in self.symmetry):
            return self

        return self.updated_copy(**self._symmetry_update_dict, deep=False, validate=False)

    @property
    def symmetry_expanded_copy(self) -> AbstractFieldData:
        """Create a copy of the :class:`.ElectromagneticSurfaceFieldData` with fields expanded based on symmetry.

        Returns
        -------
        :class:`ElectromagneticSurfaceFieldData`
            A data object with the symmetry expanded fields.
        """

        if all(sym == 0 for sym in self.symmetry):
            return self.copy()

        return self.copy(update=self._symmetry_update_dict)

    @property
    def _symmetry_update_dict(self) -> dict:
        """Dictionary of data fields to create data with expanded symmetry."""

        eigvals = self.symmetry_eigenvalues

        h_symmetry = [
            xr.DataArray(
                [eigvals["Hx"](dim), eigvals["Hy"](dim), eigvals["Hz"](dim)],
                coords={"axis": [0, 1, 2]},
            )
            * self.symmetry[dim]
            for dim in range(3)
        ]

        e_symmetry = [
            xr.DataArray(
                [eigvals["Ex"](dim), eigvals["Ey"](dim), eigvals["Ez"](dim)],
                coords={"axis": [0, 1, 2]},
            )
            * self.symmetry[dim]
            for dim in range(3)
        ]

        # normal field is always even under symmetry
        n_symmetry = [
            xr.DataArray(
                [eigvals["Ex"](dim), eigvals["Ey"](dim), eigvals["Ez"](dim)],
                coords={"axis": [0, 1, 2]},
            )
            for dim in range(3)
        ]

        updated_dict: dict[str, Any] = {}
        if self.E is not None:
            updated_dict["E"] = self._symmetry_expanded_copy_base(self.E, e_symmetry)
        if self.H is not None:
            updated_dict["H"] = self._symmetry_expanded_copy_base(self.H, h_symmetry)

        updated_dict["normal"] = self._symmetry_expanded_copy_base(self.normal, n_symmetry)
        updated_dict.update({"symmetry": (0, 0, 0), "symmetry_center": (0, 0, 0)})
        return updated_dict

    def _check_fields_stored(self, components: list[str]) -> None:
        """Check that all requested field components are stored in the data."""
        missing_comps = [comp for comp in components if comp not in self.field_components.keys()]
        if len(missing_comps) > 0:
            raise DataError(
                f"Field components {missing_comps} not included in this data object. Use "
                "the 'fields' argument of a field monitor to select which components are stored."
            )


class SurfaceFieldData(ElectromagneticSurfaceFieldData):
    """
    Data associated with a :class:`.SurfaceFieldMonitor`: E and H fields on a surface.

    Example
    -------
    >>> from tidy3d import PointDataArray, IndexedSurfaceFieldDataArray, TriangularSurfaceDataset, CellDataArray
    >>> import tidy3d as td
    >>> old_logging_level = td.config.logging.level
    >>> td.config.logging.level = "ERROR"
    >>> points = PointDataArray([[0, 0, 0], [0, 1, 0], [1, 1, 1]], dims=["index", "axis"])
    >>> cells = CellDataArray([[0, 1, 2]], dims=["cell_index", "vertex_index"])
    >>> values = PointDataArray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dims=["index", "axis"])
    >>> field_values = IndexedSurfaceFieldDataArray(np.ones((3, 1, 3, 1)) + 0j, coords={"index": [0, 1, 2], "side": ["outside"],"axis": [0, 1, 2], "f": [1e10]})
    >>> field = TriangularSurfaceDataset(points=points, cells=cells, values=field_values)
    >>> normal = TriangularSurfaceDataset(points=points, cells=cells, values=values)
    >>> monitor = SurfaceFieldMonitor(
    ...     size=(2,4,6), freqs=[1e10], name='field', fields=['E', 'H']
    ... )
    >>> data = SurfaceFieldData(monitor=monitor, E=field, H=field, normal=normal)
    >>> td.config.logging.level = old_logging_level
    """

    monitor: SurfaceFieldMonitor = Field(
        ..., title="Monitor", description="Frequency-domain field monitor associated with the data."
    )

    @property
    def poynting(self) -> TriangularSurfaceDataset:
        """Time-averaged Poynting vector for frequency-domain data."""

        self._check_fields_stored(["E", "H"])
        e_field = self.E
        h_field = self.H

        poynting = e_field.updated_copy(
            values=0.5 * np.real(xr.cross(e_field.values, np.conj(h_field.values), dim="axis"))
        )

        return poynting

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> SurfaceFieldData:
        """Return copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        src_amps = FreqDataArray(
            source_spectrum_fn(self.monitor.freqs), coords={"f": list(self.monitor.freqs)}
        )
        for field_name, field_data in self.field_components.items():
            fields_norm[field_name] = field_data.updated_copy(
                values=(field_data.values / src_amps).astype(field_data.values.dtype)
            )

        return self.copy(update=fields_norm)


class SurfaceFieldTimeData(ElectromagneticSurfaceFieldData):
    """

    Example
    -------
    >>> from tidy3d import PointDataArray, IndexedSurfaceFieldTimeDataArray, TriangularSurfaceDataset, CellDataArray
    >>> import tidy3d as td
    >>> old_logging_level = td.config.logging.level
    >>> td.config.logging.level = "ERROR"
    >>> points = PointDataArray([[0, 0, 0], [0, 1, 0], [1, 1, 1]], dims=["index", "axis"])
    >>> cells = CellDataArray([[0, 1, 2]], dims=["cell_index", "vertex_index"])
    >>> values = PointDataArray([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dims=["index", "axis"])
    >>> field_values = IndexedSurfaceFieldTimeDataArray(np.ones((3, 1, 3, 1)) + 0j, coords={"index": [0, 1, 2], "side": ["outside"],"axis": [0, 1, 2], "t": [1e-9]})
    >>> field = TriangularSurfaceDataset(points=points, cells=cells, values=field_values)
    >>> normal = TriangularSurfaceDataset(points=points, cells=cells, values=values)
    >>> monitor = SurfaceFieldTimeMonitor(
    ...     size=(2,4,6), interval=100, name='field', fields=['E', 'H']
    ... )
    >>> data = SurfaceFieldTimeData(monitor=monitor, E=field, H=field, normal=normal)
    >>> td.config.logging.level = old_logging_level
    """

    monitor: SurfaceFieldTimeMonitor = Field(
        ..., title="Monitor", description="Time-domain field monitor associated with the data."
    )

    @property
    def poynting(self) -> TriangularSurfaceDataset:
        """Poynting vector for time-domain data."""

        self._check_fields_stored(["E", "H"])
        e_field = self.E
        h_field = self.H

        poynting = e_field.updated_copy(
            values=xr.cross(np.real(e_field.values), np.real(h_field.values), dim="axis")
        )

        return poynting

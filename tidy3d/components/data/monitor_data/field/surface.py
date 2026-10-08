from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any

import autograd.numpy as np
import xarray as xr
from flexcompute.core._migration.em.numerical.raw import power_loss as loss_numerics
from pydantic import Field

from tidy3d.components.base import TYPE_TAG_STR
from tidy3d.components.base_sim.data.monitor_data import AbstractUnstructuredMonitorData
from tidy3d.components.data.data_array import FreqDataArray
from tidy3d.components.data.dataset import ElectromagneticSurfaceFieldDataset
from tidy3d.components.data.monitor_data.base import MonitorData
from tidy3d.components.data.structure_loss import (
    _check_owner_range,
    _loses_power_at_surface,
)
from tidy3d.components.data.unstructured.surface import TriangularSurfaceDataset
from tidy3d.components.monitor import (
    SurfaceFieldMonitor,
    SurfaceFieldStructureMonitor,
    SurfaceFieldTimeMonitor,
)
from tidy3d.components.validators import enforce_monitor_fields_present
from tidy3d.exceptions import AdjointError, DataError

if TYPE_CHECKING:
    from collections.abc import Callable

    from tidy3d.components.data.monitor_data.base import AbstractFieldData
    from tidy3d.components.simulation import Simulation
    from tidy3d.components.source.base import Source


class ElectromagneticSurfaceFieldData(
    MonitorData, AbstractUnstructuredMonitorData, ElectromagneticSurfaceFieldDataset, ABC
):
    """Collection of vector fields on a surface with some symmetry properties."""

    monitor: SurfaceFieldMonitor | SurfaceFieldStructureMonitor | SurfaceFieldTimeMonitor = Field(
        discriminator=TYPE_TAG_STR
    )

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


class SurfaceFieldStructureData(SurfaceFieldData):
    """
    Data associated with a :class:`.SurfaceFieldStructureMonitor`: the H field (and the E field
    when recorded), the interface normal, and per-vertex structure ownership on a triangulated
    PEC/lossy-metal surface.

    Notes
    -----

        The stored primitives (``E``, ``H``, ``normal``, ``structure_index``) are recorded on the
        triangulated surface mesh. The derived tangential-magnetic-field magnitude
        (:attr:`h_tan_squared`) and surface loss density (:meth:`surface_loss_density`) are
        computed on demand from the raw ``self`` arrays.
    """

    monitor: SurfaceFieldStructureMonitor = Field(
        ...,
        title="Monitor",
        description="Frequency-domain field-structure monitor associated with the data.",
    )

    structure_index: TriangularSurfaceDataset = Field(
        ...,
        title="Structure Index",
        description="Per-vertex owning-structure index on the triangulated surface (-1 for the "
        "background medium, i for simulation.structures[i]).",
    )

    @property
    def _symmetry_update_dict(self) -> dict:
        """Extend the base E/H/normal symmetry expansion with ``structure_index``, always even
        under symmetry (which structure owns a point doesn't change under reflection).

        Unlike ``normal`` (a vector field with an ``axis`` dimension, whose per-component
        symmetry factor is genuinely axis-tagged), ``structure_index`` is a scalar per-vertex
        value with no ``axis`` dimension. ``reflect()`` requires an ``xr.DataArray`` symmetry
        factor's dims to be a subset of the value's dims, so a plain scalar ``1`` per simulation
        axis is used instead -- not an axis-tagged eigenvalue array copied from ``normal``.
        """
        updated_dict = super()._symmetry_update_dict
        updated_dict["structure_index"] = self._symmetry_expanded_copy_base(
            self.structure_index, (1, 1, 1)
        )
        return updated_dict

    def _make_adjoint_sources(self, dataset_names: list[str], fwidth: float) -> list[Source]:
        """Objectives through this diagnostic monitor's data are unsupported: fail loudly.

        The inherited default returns no sources, which would silently produce zero-looking
        gradients instead of an error -- same failure mode :meth:`FieldStructureData.
        _make_adjoint_sources` guards against for the volumetric diagnostic monitor.
        """
        raise AdjointError(
            f"Objective functions depending on data of 'SurfaceFieldStructureMonitor' "
            f"'{self.monitor.name}' are not supported: it is a diagnostic monitor with no "
            "adjoint implementation. Base the objective on a supported monitor type instead "
            "(e.g. 'FieldMonitor'). The 'SurfaceFieldStructureMonitor' can still be carried in "
            "the simulation as long as the objective does not depend on its data."
        )

    @property
    def h_tan_squared(self) -> TriangularSurfaceDataset:
        """``|H_tan|^2`` at each vertex: squared magnitude of ``H``'s component tangential to the
        interface normal, collapsing both tangential directions into one scalar."""
        self._check_fields_stored(["H"])
        h = self.H.values.sel(side="outside")
        n = self.normal.values
        return self.normal.updated_copy(values=loss_numerics.h_tan_squared(h, n, dim="axis"))

    def surface_loss_density(self, sim: Simulation) -> TriangularSurfaceDataset:
        """Surface loss density ``SLD = 0.5 * Re(Z_s) * |H_tan|^2`` [W/um^2] at each vertex.

        ``Re(Z_s)`` is looked up per vertex from ``sim.structures[structure_index].medium``
        (``sim`` is the :class:`.Simulation` the data was recorded against; it is not stored on
        this data object, and the lookup uses each medium's ``surface_impedance()``). A vertex owned
        by the background medium (``structure_index == -1``, ``Simulation.medium`` -- e.g. a PEC-
        like surface with no explicit ``Structure``, such as one facet of an infinite lossy-metal
        half-space) looks up ``sim.medium`` instead, the same as any other owner; vertices owned
        by a lossless medium (no ``surface_impedance``, e.g. :class:`.PECMedium`) get
        ``Re(Z_s) = 0`` either way. Vertices owned by a medium the solver did not model as a
        surface, such as a :class:`.LossyMetalMedium` solved as a conductor (``subpixel=False``,
        ``Staircasing`` or ``VolumetricAveraging``), are NaN: that medium loses power in its
        volume, which a :class:`.FieldStructureMonitor` records.

        Owners are rounded and range-checked with the same ``_check_owner_range`` helper that
        ``per_structure_loss`` uses, so a non-integer owner selects the same structure, and an
        out-of-range owner raises the same error, whether this method is called directly or
        through ``per_structure_loss``.
        """
        h_tan_sq = self.h_tan_squared
        owners = _check_owner_range(
            self.structure_index.values.values, len(sim.structures), "structure_index"
        )
        freqs = np.array(self.monitor.freqs)

        re_z_s_per_vertex = np.zeros((owners.size, freqs.size))
        for owner in np.unique(owners):
            medium = sim.medium if owner < 0 else sim.structures[int(owner)].medium
            surface_impedance = getattr(medium, "surface_impedance", None)
            if not _loses_power_at_surface(medium, sim):
                re_z_s = np.full(freqs.size, np.nan)
            elif surface_impedance is None:
                re_z_s = np.zeros(freqs.size)
            else:
                re_z_s = surface_impedance(freqs).real
            re_z_s_per_vertex[owners == owner, :] = re_z_s

        h_tan = h_tan_sq.values
        re_z_s = xr.DataArray(
            re_z_s_per_vertex,
            dims=["index", "f"],
            coords={"index": h_tan.coords["index"], "f": freqs},
        )
        sld = loss_numerics.surface_loss_density(re_z_s, h_tan)
        return h_tan_sq.updated_copy(values=sld)


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

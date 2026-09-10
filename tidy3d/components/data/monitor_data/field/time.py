from __future__ import annotations

from typing import TYPE_CHECKING

import autograd.numpy as np
from flexcompute.core._migration.em.numerical.raw import field_data as field_data_numerics
from pydantic import Field

from tidy3d.components.base import cached_property
from tidy3d.components.data.data_array import FluxTimeDataArray
from tidy3d.components.data.dataset import (
    AuxFieldTimeDataset,
    FieldTimeDataset,
)
from tidy3d.components.data.monitor_data.base import AbstractFieldData
from tidy3d.components.monitor import (
    AuxFieldTimeMonitor,
    FieldTimeMonitor,
)
from tidy3d.components.validators import enforce_monitor_fields_present
from tidy3d.exceptions import DataError

from .electromagnetic import ElectromagneticFieldData

if TYPE_CHECKING:
    import xarray as xr

    from tidy3d.components.data.data_array import (
        DataArray,
        FluxDataArray,
        FreqModeDataArray,
        ScalarFieldTimeDataArray,
    )


class FieldTimeData(FieldTimeDataset, ElectromagneticFieldData):
    """
    Data associated with a :class:`.FieldTimeMonitor`: scalar components of E and H fields.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

    Example
    -------
    >>> from tidy3d import Grid, ScalarFieldTimeDataArray
    >>> from tidy3d.components.grid.grid import Coords
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], t=t)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> scalar_field = ScalarFieldTimeDataArray(np.random.random((2,3,4,3)), coords=coords)
    >>> monitor = FieldTimeMonitor(
    ...     size=(2,4,6), interval=100, name='field', fields=['Ex', 'Hz'], colocate=True
    ... )
    >>> data = FieldTimeData(monitor=monitor, Ex=scalar_field, Hz=scalar_field, grid_expanded=grid)
    """

    monitor: FieldTimeMonitor = Field(
        title="Monitor",
        description="Time-domain field monitor associated with the data.",
    )

    _contains_monitor_fields = enforce_monitor_fields_present()

    @property
    def poynting(self) -> ScalarFieldTimeDataArray:
        """Instantaneous Poynting vector for time-domain data associated to a 2D monitor, projected
        to the direction normal to the monitor plane."""

        # Tangential fields are ordered as E1, E2, H1, H2
        tan_fields = self._colocated_tangential_fields
        dim1, dim2 = self._tangential_dims
        e_x_h = np.real(tan_fields["E" + dim1]) * np.real(tan_fields["H" + dim2])
        e_x_h -= np.real(tan_fields["E" + dim2]) * np.real(tan_fields["H" + dim1])
        return e_x_h

    def _compute_flux(self) -> FluxTimeDataArray:
        """Compute instantaneous flux."""
        dim1, dim2 = self._tangential_dims
        tangential_dims = self._tangential_dims

        if getattr(self.monitor, "use_colocated_integration", self.monitor.colocate):
            fields = self._colocated_tangential_fields
            dS = self._diff_area.to_numpy()
            dS_numpy = (dS, dS)
        else:
            fields = self._tangential_fields
            dS_EuHv, dS_EvHu, _, _ = self._diff_area_at_yee_positions(
                truncate_to_monitor_bounds=True
            )
            dS_numpy = (dS_EuHv.to_numpy(), dS_EvHu.to_numpy())

        # Put spatial dims last: (..., u, v)
        Eu = np.real(fields["E" + dim1].transpose(..., *tangential_dims).to_numpy())
        Ev = np.real(fields["E" + dim2].transpose(..., *tangential_dims).to_numpy())
        Hu = np.real(fields["H" + dim1].transpose(..., *tangential_dims).to_numpy())
        Hv = np.real(fields["H" + dim2].transpose(..., *tangential_dims).to_numpy())

        flux_result = field_data_numerics.instantaneous_power_flow((Eu, Ev), (Hu, Hv), dS_numpy)

        return FluxTimeDataArray(
            flux_result, coords={"t": fields["E" + dim1].coords["t"].to_numpy()}
        )

    @cached_property
    def flux(self) -> FluxTimeDataArray:
        """Flux for data corresponding to a 2D monitor."""
        return self._compute_flux()

    def _compute_complex_flux(self) -> FluxDataArray | FreqModeDataArray:
        """Complex flux is not defined for time-domain data."""
        raise DataError("Complex power flow is not defined for time-domain data.")

    @cached_property
    def complex_flux(self) -> DataArray:
        """Complex flux is not defined for time-domain data."""
        raise DataError("Complex power flow is not defined for time-domain data.")

    def dot(
        self,
        field_data: ElectromagneticFieldData,
        conjugate: bool = True,
        bidirectional: bool = True,
    ) -> xr.DataArray:
        """Inner product is not defined for time-domain data."""
        raise DataError("Inner product is not defined for time-domain data.")

    def outer_dot(
        self,
        field_data: ElectromagneticFieldData,
        conjugate: bool = True,
        bidirectional: bool = True,
    ) -> xr.DataArray:
        """Outer dot product is not defined for time-domain data."""
        raise DataError("Outer dot product is not defined for time-domain data.")

    @property
    def time_reversed_copy(self) -> FieldTimeData:
        """Make a copy of the data with time-reversed fields. The sign of the magnetic fields is
        flipped, and the data is reversed along the ``t`` dimension, such that for a given field,
        ``field[t_beg + t] -> field[t_end - t]``, where ``t_beg`` and ``t_end`` are the first and
        last coordinates along the ``t`` dimension.
        """
        new_data = {}
        for comp, field in self.field_components.items():
            if comp[0] == "H":
                new_data[comp] = -field
            else:
                new_data[comp] = field
            # Reverse time coordinates
            new_data[comp] = new_data[comp].assign_coords({"t": field.t[::-1]}).sortby("t")
        return self.copy(deep=False, update=new_data)


class AuxFieldTimeData(AuxFieldTimeDataset, AbstractFieldData):
    """
    Data associated with a :class:`.AuxFieldTimeMonitor`: scalar components of aux fields.

    Notes
    -----

        The data is stored as a `DataArray <https://docs.xarray.dev/en/stable/generated/xarray.DataArray.html>`_
        object using the `xarray <https://docs.xarray.dev/en/stable/index.html>`_ package.

    Example
    -------
    >>> from tidy3d import Grid, ScalarFieldTimeDataArray
    >>> from tidy3d.components.grid.grid import Coords
    >>> x = [-1,1,3]
    >>> y = [-2,0,2,4]
    >>> z = [-3,-1,1,3,5]
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(x=x[:-1], y=y[:-1], z=z[:-1], t=t)
    >>> grid = Grid(boundaries=Coords(x=x, y=y, z=z))
    >>> scalar_field = ScalarFieldTimeDataArray(np.random.random((2,3,4,3)), coords=coords)
    >>> monitor = AuxFieldTimeMonitor(
    ...     size=(2,4,6), interval=100, name='field', fields=['Nfx'], colocate=True
    ... )
    >>> data = AuxFieldTimeData(monitor=monitor, Nfx=scalar_field, grid_expanded=grid)
    """

    monitor: AuxFieldTimeMonitor = Field(
        title="Monitor",
        description="Time-domain auxiliary field monitor associated with the data.",
    )

    _contains_monitor_fields = enforce_monitor_fields_present()

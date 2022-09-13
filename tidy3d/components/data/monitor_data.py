""" Monitor Level Data, store the DataArrays associated with a single monitor."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Dict, Tuple, Callable
import xarray as xr
import numpy as np
import pydantic as pd

from ..base import TYPE_TAG_STR, Tidy3dBaseModel
from ..types import Axis, Coordinate, Symmetry
from ..grid.grid import Grid
from ..validators import enforce_monitor_fields_present
from ..monitor import MonitorType, FieldMonitor, FieldTimeMonitor, ModeSolverMonitor
from ..monitor import ModeMonitor, FluxMonitor, FluxTimeMonitor, PermittivityMonitor
from .dataset import Dataset, FieldData, FieldTimeData, ModeSolverData, PermittivityData, ModeData
from .dataset import FluxData, FluxTimeData
from ...log import DataError


class MonitorData(Tidy3dBaseModel, ABC):
    """Abstract base class of objects that store data pertaining to a single :class:`.monitor`."""

    monitor: MonitorType = pd.Field(
        ...,
        title="Monitor",
        description="Monitor associated with the data.",
        descriminator=TYPE_TAG_STR,
    )

    dataset: Dataset = pd.Field(
        ...,
        title="Dataset",
        description="Dataset corresponding to the monitor.",
        descriminator=TYPE_TAG_STR,
    )

    # pylint:disable=unused-argument
    def apply_symmetry(
        self,
        symmetry: Tuple[Symmetry, Symmetry, Symmetry],
        symmetry_center: Coordinate,
        grid_expanded: Grid,
    ) -> MonitorData:
        """Return copy of self with symmetry applied."""
        return self

    def normalize(
        self, source_spectrum_fn: Callable[[float], complex]  # pylint:disable=unused-argument
    ) -> MonitorData:
        """Return copy of self after normalization is applied using source spectrum function."""
        return self.copy()


class AbstractFieldMonitorData(MonitorData, ABC):
    """Collection of scalar fields with some symmetry properties."""

    monitor: Union[FieldMonitor, FieldTimeMonitor, PermittivityMonitor, ModeSolverMonitor]

    def apply_symmetry(  # pylint:disable=too-many-locals
        self,
        symmetry: Tuple[Symmetry, Symmetry, Symmetry],
        symmetry_center: Coordinate,
        grid_expanded: Grid,
    ) -> AbstractFieldMonitorData:
        """Create a copy of the :class:`.AbstractFieldData` with the fields expanded based on
        symmetry, if any.

        Returns
        -------
        :class:`AbstractFieldData`
            A data object with the symmetry expanded fields.
        """

        if all(sym == 0 for sym in symmetry):
            return self.copy()

        new_fields = {}

        for field_name, scalar_data in self.dataset.field_components.items():

            grid_key = self.dataset.grid_locations[field_name]
            eigenval_fn = self.dataset.symmetry_eigenvalues[field_name]

            # get grid locations for this field component on the expanded grid
            grid_locations = grid_expanded[grid_key]

            for sym_dim, (sym_val, sym_center) in enumerate(zip(symmetry, symmetry_center)):

                dim_name = "xyz"[sym_dim]

                # Continue if no symmetry along this dimension
                if sym_val == 0:
                    continue

                # Get coordinates for this field component on the expanded grid
                coords = grid_locations.to_list[sym_dim]

                # Get indexes of coords that lie on the left of the symmetry center
                flip_inds = np.where(coords < sym_center)[0]

                # Get the symmetric coordinates on the right
                coords_interp = np.copy(coords)
                coords_interp[flip_inds] = 2 * sym_center - coords[flip_inds]

                # Interpolate. There generally shouldn't be values out of bounds except potentially
                # when handling modes, in which case they should be at the boundary and close to 0.
                data_array = scalar_data.data
                data_array = data_array.sel({dim_name: coords_interp}, method="nearest")
                data_array = data_array.assign_coords({dim_name: coords})

                # apply the symmetry eigenvalue (if defined) to the flipped values
                if eigenval_fn is not None:
                    sym_eigenvalue = eigenval_fn(sym_dim)
                    data_array[{dim_name: flip_inds}] *= sym_val * sym_eigenvalue

            # assign the final scalar data to the new_fields
            new_fields[field_name] = scalar_data.copy(update=dict(data=data_array))

        new_dataset = self.dataset.copy(update=new_fields)
        return self.copy(update=dict(dataset=new_dataset))


class FieldMonitorData(AbstractFieldMonitorData):
    """Data associated with a :class:`.FieldMonitor`: scalar components of E and H fields.

    Example
    -------
    >>> from .data_array import ScalarFieldDataArray
    >>> x = [-1,1]
    >>> y = [-2,0,2]
    >>> z = [-3,-1,1,3]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> data_array = xr.DataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> scalar_field = ScalarFieldDataArray(data=data_array)
    >>> field_data = FieldData(Ex=scalar_field, Hz=scalar_field)
    >>> monitor = FieldMonitor(size=(2,4,6), freqs=[2e14, 3e14], name='field', fields=['Ex', 'Hz'])
    >>> data = FieldMonitorData(monitor=monitor, dataset=field_data)
    """

    monitor: FieldMonitor
    dataset: FieldData

    _contains_monitor_fields = enforce_monitor_fields_present()

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> FieldData:
        """Return copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        for field_name, field_data in self.dataset.field_components.items():
            src_amps = source_spectrum_fn(field_data.data.f)
            new_data_array = field_data.data / src_amps
            fields_norm[field_name] = field_data.copy(update=dict(data=new_data_array))

        new_dataset = self.dataset.copy(update=fields_norm)
        return self.copy(update=dict(dataset=new_dataset))


class FieldTimeMonitorData(AbstractFieldMonitorData):
    """Data associated with a :class:`.FieldTimeMonitor`: scalar components of E and H fields.

    Example
    -------
    >>> from .data_array import ScalarFieldTimeDataArray
    >>> x = [-1,1]
    >>> y = [-2,0,2]
    >>> z = [-3,-1,1,3]
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(x=x, y=y, z=z, t=t)
    >>> data_array = xr.DataArray(np.random.random((2,3,4,3)), coords=coords)
    >>> scalar_field = ScalarFieldTimeDataArray(data=data_array)
    >>> monitor = FieldTimeMonitor(size=(2,4,6), interval=100, name='field', fields=['Ex', 'Hz'])
    >>> dataset = FieldTimeData(Ex=scalar_field, Hz=scalar_field)
    >>> data = FieldTimeMonitorData(monitor=monitor, dataset=dataset)
    """

    monitor: FieldTimeMonitor
    dataset: FieldTimeData

    _contains_monitor_fields = enforce_monitor_fields_present()


class ModeSolverMonitorData(AbstractFieldMonitorData):
    """Data associated with a :class:`.ModeSolverMonitor`: scalar components of E and H fields.

    Example
    -------
    >>> from tidy3d import ModeSpec
    >>> from .data_array import ScalarModeFieldDataArray, ModeIndexDataArray
    >>> x = [-1,1]
    >>> y = [0]
    >>> z = [-3,-1,1,3]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> field_coords = dict(x=x, y=y, z=z, f=f, mode_index=mode_index)
    >>> data_array_field = xr.DataArray((1+1j)*np.random.random((2,1,4,2,5)), coords=field_coords)
    >>> sf = ScalarModeFieldDataArray(data=data_array_field)
    >>> index_coords = dict(f=f, mode_index=mode_index)
    >>> data_array_index = xr.DataArray((1+1j) * np.random.random((2,5)), coords=index_coords)
    >>> index_data = ModeIndexDataArray(data=data_array_index)
    >>> dataset = ModeSolverData(Ex=sf, Ey=sf, Ez=sf, Hx=sf, Hy=sf, Hz=sf, n_complex=index_data)
    >>> monitor = ModeSolverMonitor(size=(2,0,6),
    ...    freqs=[2e14, 3e14],
    ...    mode_spec=ModeSpec(num_modes=5),
    ...    name='ms',
    ... )
    >>> data = ModeSolverMonitorData(monitor=monitor, dataset=dataset)
    """

    monitor: ModeSolverMonitor
    dataset: ModeSolverData

    def plot_field(self, *args, **kwargs):
        """Warn user to use the :class:`.ModeSolver` ``plot_field`` function now."""
        raise DeprecationWarning(
            "The 'plot_field()' method was moved to the 'ModeSolver' object."
            "Once the 'ModeSolver' is contructed, one may call '.plot_field()' on the object and "
            "the modes will be computed and displayed with 'Simulation' overlay."
        )


class PermittivityMonitorData(AbstractFieldMonitorData):
    """Data for a :class:`.PermittivityMonitor`: diagonal components of the permittivity tensor.

    Example
    -------
    >>> from .data_array import ScalarFieldDataArray
    >>> x = [-1,1]
    >>> y = [-2,0,2]
    >>> z = [-3,-1,1,3]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> data_array = xr.DataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> scalar_field = ScalarFieldDataArray(data=data_array)
    >>> dataset = PermittivityData(eps_xx=scalar_field, eps_yy=scalar_field, eps_zz=scalar_field)
    >>> monitor = PermittivityMonitor(size=(2,4,6), freqs=[2e14, 3e14], name='eps')
    >>> data = PermittivityMonitorData(monitor=monitor, dataset=dataset)
    """

    monitor: PermittivityMonitor
    dataset: PermittivityData


class ModeMonitorData(MonitorData):
    """Data associated with a :class:`.ModeMonitor`: modal amplitudes and propagation indices.

    Example
    -------
    >>> from tidy3d import ModeSpec
    >>> from .data_array import ModeIndexDataArray, ModeAmpsDataArray
    >>> direction = ["+", "-"]
    >>> f = [1e14, 2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> index_coords = dict(f=f, mode_index=mode_index)
    >>> data_array_index = xr.DataArray((1+1j) * np.random.random((3, 5)), coords=index_coords)
    >>> index_data = ModeIndexDataArray(data=data_array_index)
    >>> amp_coords = dict(direction=direction, f=f, mode_index=mode_index)
    >>> data_array_amp = xr.DataArray((1+1j) * np.random.random((2, 3, 5)), coords=amp_coords)
    >>> amp_data = ModeAmpsDataArray(data=data_array_amp)
    >>> dataset = ModeData(amps=amp_data, n_complex=index_data)
    >>> monitor = ModeMonitor(
    ...    size=(2,0,6),
    ...    freqs=[2e14, 3e14],
    ...    mode_spec=ModeSpec(num_modes=5),
    ...    name='mode',
    ... )
    >>> data = ModeMonitorData(monitor=monitor, dataset=dataset)
    """

    monitor: ModeMonitor
    dataset: ModeData

    def normalize(self, source_spectrum_fn) -> ModeData:
        """Return copy of self after normalization is applied using source spectrum function."""
        if self.dataset.amps is None:
            raise DataError("ModeData contains no amp data, can't normalize.")
        source_freq_amps = source_spectrum_fn(self.dataset.amps.data.f)[None, :, None]

        new_data_array = self.dataset.amps.data / source_freq_amps
        new_amps_data = self.dataset.amps.copy(update=dict(data=new_data_array))
        new_datset = self.dataset.copy(update=dict(amps=new_amps_data))
        return self.copy(update=dict(dataset=new_datset))

class FluxMonitorData(MonitorData):
    """Data associated with a :class:`.FluxMonitor`: flux data in the frequency-domain.

    Example
    -------
    >>> from .data_array import FluxDataArray
    >>> f = [2e14, 3e14]
    >>> coords = dict(f=f)
    >>> data_array = xr.DataArray(np.random.random(2), coords=coords)
    >>> flux_data = FluxDataArray(data=data_array)
    >>> dataset = FluxData(flux=flux_data)
    >>> monitor = FluxMonitor(size=(2,0,6), freqs=[2e14, 3e14], name='flux')
    >>> data = FluxMonitorData(monitor=monitor, dataset=dataset)
    """

    monitor: FluxMonitor
    dataset: FluxData

    def normalize(self, source_spectrum_fn) -> FluxData:
        """Return copy of self after normalization is applied using source spectrum function."""
        source_freq_amps = source_spectrum_fn(self.dataset.flux.data.f)
        source_power = abs(source_freq_amps) ** 2
        new_data_array = self.dataset.flux.data / source_power
        new_flux_data = self.dataset.flux.copy(update=dict(data=new_data_array))
        new_datset = self.dataset.copy(update=dict(flux=new_flux_data))
        return self.copy(update=dict(dataset=new_datset))


class FluxTimeMonitorData(MonitorData):
    """Data associated with a :class:`.FluxTimeMonitor`: flux data in the time-domain.

    Example
    -------
    >>> from .data_array import FluxTimeDataArray
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(t=t)
    >>> data_array = xr.DataArray(np.random.random(3), coords=coords)
    >>> flux_data = FluxTimeDataArray(data=data_array)
    >>> dataset = FluxTimeData(flux=flux_data)
    >>> monitor = FluxTimeMonitor(size=(2,0,6), interval=100, name='flux_time')
    >>> data = FluxTimeMonitorData(monitor=monitor, dataset=dataset)
    """

    monitor: FluxTimeMonitor
    dataset: FluxTimeData


MonitorDataTypes = (
    FieldMonitorData,
    FieldTimeMonitorData,
    PermittivityMonitorData,
    ModeSolverMonitorData,
    ModeMonitorData,
    FluxMonitorData,
    FluxTimeMonitorData,
)

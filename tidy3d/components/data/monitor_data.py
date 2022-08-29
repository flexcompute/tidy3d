""" Monitor Level Data, store the DataArrays associated with a single monitor."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Dict, Tuple, Callable
import xarray as xr
import numpy as np
import pydantic as pd

from ..base import TYPE_TAG_STR, Tidy3dBaseModel
from ..types import Axis, Coordinate
from ..boundary import Symmetry
from ..grid import Grid
from ..validators import enforce_monitor_fields_present
from ..monitor import MonitorType, FieldMonitor, FieldTimeMonitor, ModeSolverMonitor
from ..monitor import ModeMonitor, FluxMonitor, FluxTimeMonitor, PermittivityMonitor
from ...log import DataError

from .data_array import ScalarFieldDataArray, ScalarFieldTimeDataArray, ScalarModeFieldDataArray
from .data_array import FluxTimeDataArray, FluxDataArray, ModeIndexDataArray, ModeAmpsDataArray
from .data_array import DataArray


class MonitorData(Tidy3dBaseModel, ABC):
    """Abstract base class of objects that store data pertaining to a single :class:`.monitor`."""

    monitor: MonitorType = pd.Field(
        ...,
        title="Monitor",
        description="Monitor associated with the data.",
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


class AbstractFieldData(MonitorData, ABC):
    """Collection of scalar fields with some symmetry properties."""

    monitor: Union[FieldMonitor, FieldTimeMonitor, PermittivityMonitor, ModeSolverMonitor]

    @property
    @abstractmethod
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to thier associated data."""

    @property
    @abstractmethod
    def grid_locations(self) -> Dict[str, str]:
        """Maps field components to the string key of their grid locations on the yee lattice."""

    @property
    @abstractmethod
    def symmetry_eigenvalues(self) -> Dict[str, Callable[[Axis], float]]:
        """Maps field components to their (positive) symmetry eigenvalues."""

    def apply_symmetry(  # pylint:disable=too-many-locals
        self,
        symmetry: Tuple[Symmetry, Symmetry, Symmetry],
        symmetry_center: Coordinate,
        grid_expanded: Grid,
    ) -> AbstractFieldData:
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

        for field_name, scalar_data in self.field_components.items():

            grid_key = self.grid_locations[field_name]
            eigenval_fn = self.symmetry_eigenvalues[field_name]

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
                scalar_data = scalar_data.sel({dim_name: coords_interp}, method="nearest")
                scalar_data = scalar_data.assign_coords({dim_name: coords})

                # apply the symmetry eigenvalue (if defined) to the flipped values
                if eigenval_fn is not None:
                    sym_eigenvalue = eigenval_fn(sym_dim)
                    scalar_data[{dim_name: flip_inds}] *= sym_val * sym_eigenvalue

            # assign the final scalar data to the new_fields
            new_fields[field_name] = scalar_data

        return self.copy(update=new_fields)

    def colocate(self, x=None, y=None, z=None) -> xr.Dataset:
        """colocate all of the data at a set of x, y, z coordinates.

        Parameters
        ----------
        x : Optional[array-like] = None
            x coordinates of locations.
            If not supplied, does not try to colocate on this dimension.
        y : Optional[array-like] = None
            y coordinates of locations.
            If not supplied, does not try to colocate on this dimension.
        z : Optional[array-like] = None
            z coordinates of locations.
            If not supplied, does not try to colocate on this dimension.

        Returns
        -------
        xr.Dataset
            Dataset containing all fields at the same spatial locations.
            For more details refer to `xarray's Documentaton <https://tinyurl.com/cyca3krz>`_.

        Note
        ----
        For many operations (such as flux calculations and plotting),
        it is important that the fields are colocated at the same spatial locations.
        Be sure to apply this method to your field data in those cases.
        """

        # convert supplied coordinates to array and assign string mapping to them
        supplied_coord_map = {k: np.array(v) for k, v in zip("xyz", (x, y, z)) if v is not None}

        # dict of data arrays to combine in dataset and return
        centered_fields = {}

        # loop through field components
        for field_name, field_data in self.field_components.items():

            # loop through x, y, z dimensions and raise an error if only one element along dim
            for coord_name, coords_supplied in supplied_coord_map.items():
                coord_data = field_data.coords[coord_name]
                if coord_data.size == 1:
                    raise DataError(
                        f"colocate given {coord_name}={coords_supplied}, but "
                        f"data only has one coordinate at {coord_name}={coord_data.values[0]}. "
                        "Therefore, can't colocate along this dimension. "
                        f"supply {coord_name}=None to skip it."
                    )

            centered_fields[field_name] = field_data.interp(
                **supplied_coord_map, kwargs={"bounds_error": True}
            )

        # combine all centered fields in a dataset
        return xr.Dataset(centered_fields)


class ElectromagneticFieldData(AbstractFieldData, ABC):
    """Stores a collection of E and H fields with x, y, z components."""

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to thier associated data."""
        # pylint:disable=no-member
        return {field: getattr(self, field) for field in self.monitor.fields}

    @property
    def grid_locations(self) -> Dict[str, str]:
        """Maps field components to the string key of their grid locations on the yee lattice."""
        return dict(Ex="Ex", Ey="Ey", Ez="Ez", Hx="Hx", Hy="Hy", Hz="Hz")

    @property
    def symmetry_eigenvalues(self) -> Dict[str, Callable[[Axis], float]]:
        """Maps field components to their (positive) symmetry eigenvalues."""

        return dict(
            Ex=lambda dim: -1 if (dim == 0) else +1,
            Ey=lambda dim: -1 if (dim == 1) else +1,
            Ez=lambda dim: -1 if (dim == 2) else +1,
            Hx=lambda dim: +1 if (dim == 0) else -1,
            Hy=lambda dim: +1 if (dim == 1) else -1,
            Hz=lambda dim: +1 if (dim == 2) else -1,
        )


class FieldData(ElectromagneticFieldData):
    """Data associated with a :class:`.FieldMonitor`: scalar components of E and H fields.

    Example
    -------
    >>> x = [-1,1]
    >>> y = [-2,0,2]
    >>> z = [-3,-1,1,3]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> scalar_field = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> monitor = FieldMonitor(size=(2,4,6), freqs=[2e14, 3e14], name='field', fields=['Ex', 'Hz'])
    >>> data = FieldData(monitor=monitor, Ex=scalar_field, Hz=scalar_field)
    """

    monitor: FieldMonitor

    Ex: ScalarFieldDataArray = pd.Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field.",
    )
    Ey: ScalarFieldDataArray = pd.Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field.",
    )
    Ez: ScalarFieldDataArray = pd.Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field.",
    )
    Hx: ScalarFieldDataArray = pd.Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field.",
    )
    Hy: ScalarFieldDataArray = pd.Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field.",
    )
    Hz: ScalarFieldDataArray = pd.Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field.",
    )

    _contains_monitor_fields = enforce_monitor_fields_present()

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> FieldData:
        """Return copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        for field_name, field_data in self.field_components.items():
            src_amps = source_spectrum_fn(field_data.f)
            fields_norm[field_name] = field_data / src_amps

        return self.copy(update=fields_norm)


class FieldTimeData(ElectromagneticFieldData):
    """Data associated with a :class:`.FieldTimeMonitor`: scalar components of E and H fields.

    Example
    -------
    >>> x = [-1,1]
    >>> y = [-2,0,2]
    >>> z = [-3,-1,1,3]
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(x=x, y=y, z=z, t=t)
    >>> scalar_field = ScalarFieldTimeDataArray(np.random.random((2,3,4,3)), coords=coords)
    >>> monitor = FieldTimeMonitor(size=(2,4,6), interval=100, name='field', fields=['Ex', 'Hz'])
    >>> data = FieldTimeData(monitor=monitor, Ex=scalar_field, Hz=scalar_field)
    """

    monitor: FieldTimeMonitor

    Ex: ScalarFieldTimeDataArray = pd.Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field.",
    )
    Ey: ScalarFieldTimeDataArray = pd.Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field.",
    )
    Ez: ScalarFieldTimeDataArray = pd.Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field.",
    )
    Hx: ScalarFieldTimeDataArray = pd.Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field.",
    )
    Hy: ScalarFieldTimeDataArray = pd.Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field.",
    )
    Hz: ScalarFieldTimeDataArray = pd.Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field.",
    )

    _contains_monitor_fields = enforce_monitor_fields_present()


class ModeSolverData(ElectromagneticFieldData):
    """Data associated with a :class:`.ModeSolverMonitor`: scalar components of E and H fields.

    Example
    -------
    >>> from tidy3d import ModeSpec
    >>> x = [-1,1]
    >>> y = [0]
    >>> z = [-3,-1,1,3]
    >>> f = [2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> field_coords = dict(x=x, y=y, z=z, f=f, mode_index=mode_index)
    >>> field = ScalarModeFieldDataArray((1+1j)*np.random.random((2,1,4,2,5)), coords=field_coords)
    >>> index_coords = dict(f=f, mode_index=mode_index)
    >>> index_data = ModeIndexDataArray((1+1j) * np.random.random((2,5)), coords=index_coords)
    >>> monitor = ModeSolverMonitor(
    ...    size=(2,0,6),
    ...    freqs=[2e14, 3e14],
    ...    mode_spec=ModeSpec(num_modes=5),
    ...    name='mode_solver',
    ... )
    >>> data = ModeSolverData(
    ...     monitor=monitor,
    ...     Ex=field,
    ...     Ey=field,
    ...     Ez=field,
    ...     Hx=field,
    ...     Hy=field,
    ...     Hz=field,
    ...     n_complex=index_data
    ... )
    """

    monitor: ModeSolverMonitor

    Ex: ScalarModeFieldDataArray = pd.Field(
        ...,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field of the mode.",
    )
    Ey: ScalarModeFieldDataArray = pd.Field(
        ...,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field of the mode.",
    )
    Ez: ScalarModeFieldDataArray = pd.Field(
        ...,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field of the mode.",
    )
    Hx: ScalarModeFieldDataArray = pd.Field(
        ...,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field of the mode.",
    )
    Hy: ScalarModeFieldDataArray = pd.Field(
        ...,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field of the mode.",
    )
    Hz: ScalarModeFieldDataArray = pd.Field(
        ...,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field of the mode.",
    )

    n_complex: ModeIndexDataArray = pd.Field(
        ...,
        title="Propagation Index",
        description="Complex-valued effective propagation constants associated with the mode.",
    )

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to thier associated data."""
        # pylint:disable=no-member
        return {field: getattr(self, field) for field in ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]}

    @property
    def n_eff(self):
        """Real part of the propagation index."""
        return self.n_complex.real

    @property
    def k_eff(self):
        """Imaginary part of the propagation index."""
        return self.n_complex.imag

    def sel_mode_index(self, mode_index: pd.NonNegativeInt) -> FieldData:
        """Return :class:`.FieldData` for the specificed mode index."""

        fields = {}
        for field_name, data in self.field_components.items():
            data = data.sel(mode_index=mode_index)
            coords = {key: val.data for key, val in data.coords.items()}
            scalar_field = ScalarFieldDataArray(data.data, coords=coords)
            fields[field_name] = scalar_field

        monitor_dict = self.monitor.dict(exclude={TYPE_TAG_STR, "mode_spec"})
        field_monitor = FieldMonitor(**monitor_dict)

        return FieldData(monitor=field_monitor, **fields)

    def plot_field(self, *args, **kwargs):
        """Warn user to use the :class:`.ModeSolver` ``plot_field`` function now."""
        raise DeprecationWarning(
            "The 'plot_field()' method was moved to the 'ModeSolver' object."
            "Once the 'ModeSolver' is contructed, one may call '.plot_field()' on the object and "
            "the modes will be computed and displayed with 'Simulation' overlay."
        )


class PermittivityData(AbstractFieldData):
    """Data for a :class:`.PermittivityMonitor`: diagonal components of the permittivity tensor.

    Example
    -------
    >>> x = [-1,1]
    >>> y = [-2,0,2]
    >>> z = [-3,-1,1,3]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> sclr_fld = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> monitor = PermittivityMonitor(size=(2,4,6), freqs=[2e14, 3e14], name='eps')
    >>> data = PermittivityData(monitor=monitor, eps_xx=sclr_fld, eps_yy=sclr_fld, eps_zz=sclr_fld)
    """

    monitor: PermittivityMonitor

    @property
    def field_components(self) -> Dict[str, ScalarFieldDataArray]:
        """Maps the field components to thier associated data."""
        return dict(eps_xx=self.eps_xx, eps_yy=self.eps_yy, eps_zz=self.eps_zz)

    @property
    def grid_locations(self) -> Dict[str, str]:
        """Maps field components to the string key of their grid locations on the yee lattice."""
        return dict(eps_xx="Ex", eps_yy="Ey", eps_zz="Ez")

    @property
    def symmetry_eigenvalues(self) -> Dict[str, Callable[[Axis], float]]:
        """Maps field components to their (positive) symmetry eigenvalues."""
        return dict(eps_xx=None, eps_yy=None, eps_zz=None)

    eps_xx: ScalarFieldDataArray = pd.Field(
        ...,
        title="Epsilon xx",
        description="Spatial distribution of the x-component of the electric field.",
    )
    eps_yy: ScalarFieldDataArray = pd.Field(
        ...,
        title="Epsilon yy",
        description="Spatial distribution of the y-component of the electric field.",
    )
    eps_zz: ScalarFieldDataArray = pd.Field(
        ...,
        title="Epsilon zz",
        description="Spatial distribution of the z-component of the electric field.",
    )


class ModeData(MonitorData):
    """Data associated with a :class:`.ModeMonitor`: modal amplitudes and propagation indices.

    Example
    -------
    >>> from tidy3d import ModeSpec
    >>> direction = ["+", "-"]
    >>> f = [1e14, 2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> index_coords = dict(f=f, mode_index=mode_index)
    >>> index_data = ModeIndexDataArray((1+1j) * np.random.random((3, 5)), coords=index_coords)
    >>> amp_coords = dict(direction=direction, f=f, mode_index=mode_index)
    >>> amp_data = ModeAmpsDataArray((1+1j) * np.random.random((2, 3, 5)), coords=amp_coords)
    >>> monitor = ModeMonitor(
    ...    size=(2,0,6),
    ...    freqs=[2e14, 3e14],
    ...    mode_spec=ModeSpec(num_modes=5),
    ...    name='mode',
    ... )
    >>> data = ModeData(monitor=monitor, amps=amp_data, n_complex=index_data)
    """

    monitor: ModeMonitor

    amps: ModeAmpsDataArray = pd.Field(
        ..., title="Amplitudes", description="Complex-valued amplitudes associated with the mode."
    )

    n_complex: ModeIndexDataArray = pd.Field(
        ...,
        title="Propagation Index",
        description="Complex-valued effective propagation constants associated with the mode.",
    )

    @property
    def n_eff(self):
        """Real part of the propagation index."""
        return self.n_complex.real

    @property
    def k_eff(self):
        """Imaginary part of the propagation index."""
        return self.n_complex.imag

    def normalize(self, source_spectrum_fn) -> ModeData:
        """Return copy of self after normalization is applied using source spectrum function."""
        if self.amps is None:
            raise DataError("ModeData contains no amp data, can't normalize.")
        source_freq_amps = source_spectrum_fn(self.amps.f)[None, :, None]
        return self.copy(update={"amps": self.amps / source_freq_amps})


class FluxData(MonitorData):
    """Data associated with a :class:`.FluxMonitor`: flux data in the frequency-domain.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> coords = dict(f=f)
    >>> flux_data = FluxDataArray(np.random.random(2), coords=coords)
    >>> monitor = FluxMonitor(size=(2,0,6), freqs=[2e14, 3e14], name='flux')
    >>> data = FluxData(monitor=monitor, flux=flux_data)
    """

    monitor: FluxMonitor
    flux: FluxDataArray

    def normalize(self, source_spectrum_fn) -> FluxData:
        """Return copy of self after normalization is applied using source spectrum function."""
        source_freq_amps = source_spectrum_fn(self.flux.f)
        source_power = abs(source_freq_amps) ** 2
        return self.copy(update={"flux": self.flux / source_power})


class FluxTimeData(MonitorData):
    """Data associated with a :class:`.FluxTimeMonitor`: flux data in the time-domain.

    Example
    -------
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(t=t)
    >>> flux_data = FluxTimeDataArray(np.random.random(3), coords=coords)
    >>> monitor = FluxTimeMonitor(size=(2,0,6), interval=100, name='flux_time')
    >>> data = FluxTimeData(monitor=monitor, flux=flux_data)
    """

    monitor: FluxTimeMonitor
    flux: FluxTimeDataArray


MonitorDataTypes = (
    FieldData,
    FieldTimeData,
    PermittivityData,
    ModeSolverData,
    ModeData,
    FluxData,
    FluxTimeData,
)

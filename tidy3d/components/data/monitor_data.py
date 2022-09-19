# pylint:disable=too-many-lines
""" Monitor Level Data, store the DataArrays associated with a single monitor."""
from __future__ import annotations

from abc import ABC
from typing import Union, Tuple, Callable, List
import numpy as np
import pydantic as pd
import xarray as xr

from .dataset import Dataset, FieldData, FieldTimeData, ModeSolverData, PermittivityData, ModeData
from .dataset import FluxData, FluxTimeData
from .dataset import Near2FarAngleData, Near2FarCartesianData, Near2FarKSpaceData

from ..base import TYPE_TAG_STR, Tidy3dBaseModel
from ..types import Coordinate, Symmetry
from ..grid.grid import Grid
from ..validators import enforce_monitor_fields_present
from ..monitor import MonitorType, FieldMonitor, FieldTimeMonitor, ModeSolverMonitor
from ..monitor import ModeMonitor, FluxMonitor, FluxTimeMonitor, PermittivityMonitor
from ..monitor import Near2FarAngleMonitor, Near2FarCartesianMonitor, Near2FarKSpaceMonitor
from ..medium import Medium
from ...log import DataError
from ...constants import C_0, ETA_0


class MonitorData(Tidy3dBaseModel, ABC):
    """Abstract base class of objects that store data pertaining to a single :class:`.monitor`."""

    monitor: MonitorType = pd.Field(
        ...,
        title="Monitor",
        description=":class:`.Monitor` associated with the data.",
        descriminator=TYPE_TAG_STR,
    )

    dataset: Dataset = pd.Field(
        ...,
        title="Dataset",
        description=":class:`.Dataset` storing the data as fields.",
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

            data_array = scalar_data.data

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

    def get_field_data_array(self, field_name: str) -> xr.DataArray:
        """Get the scalar field data associated with the string."""
        field_component = self.dataset.field_components.get(field_name)
        return None if field_component is None else field_component.data

    @property
    def Ex(self) -> xr.DataArray:
        """x-component of electric field."""
        return self.get_field_data_array("Ex")

    @property
    def Ey(self) -> xr.DataArray:
        """y-component of electric field."""
        return self.get_field_data_array("Ey")

    @property
    def Ez(self) -> xr.DataArray:
        """z-component of electric field."""
        return self.get_field_data_array("Ez")

    @property
    def Hx(self) -> xr.DataArray:
        """x-component of magnetic field."""
        return self.get_field_data_array("Hx")

    @property
    def Hy(self) -> xr.DataArray:
        """y-component of magnetic field."""
        return self.get_field_data_array("Hy")

    @property
    def Hz(self) -> xr.DataArray:
        """z-component of magnetic field."""
        return self.get_field_data_array("Hz")

    def colocate(
        self, x: List[float] = None, y: List[float] = None, z: List[float] = None
    ) -> xr.Dataset:
        """colocate all of the data at a set of x, y, z coordinates.

        Parameters
        ----------
        x : List[float] = None
            x coordinates of locations.
            If not supplied, does not try to colocate on this dimension.
        y : List[float] = None
            y coordinates of locations.
            If not supplied, does not try to colocate on this dimension.
        z : List[float] = None
            z coordinates of locations.
            If not supplied, does not try to colocate on this dimension.

        Returns
        -------
        ``xr.Dataset``
            Dataset containing all fields at the same spatial locations.
            For more details refer to `xarray's Documentaton <https://tinyurl.com/cyca3krz>`_.

        Note
        ----
        For many operations (such as flux calculations and plotting),
        it is important that the fields are colocated at the same spatial locations.
        Be sure to apply this method to your field data in those cases.
        """
        return self.dataset.colocate(x=x, y=y, z=z)


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

    @property
    def n_complex(self) -> xr.DataArray:
        """Effective index as a function of frequency and mode index."""
        return self.dataset.n_complex.data

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

    @property
    def eps_xx(self) -> xr.DataArray:
        """xx-component of relative perittivity tensor as a function of space and frequency."""
        return self.dataset.eps_xx.data

    @property
    def eps_yy(self) -> xr.DataArray:
        """yy-component of relative perittivity tensor as a function of space and frequency."""
        return self.dataset.eps_yy.data

    @property
    def eps_zz(self) -> xr.DataArray:
        """zz-component of relative perittivity tensor as a function of space and frequency."""
        return self.dataset.eps_zz.data


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

    @property
    def n_complex(self) -> xr.DataArray:
        """Effective index as a function of frequency and mode index."""
        return self.dataset.n_complex.data

    @property
    def amps(self) -> xr.DataArray:
        """Modal amplitudes as a function of direction, frequency, and mode index."""
        return self.dataset.amps.data

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

    @property
    def flux(self) -> xr.DataArray:
        """Flux as a function of frequency."""
        return self.dataset.flux.data

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

    @property
    def flux(self) -> xr.DataArray:
        """Flux as a function of time."""
        return self.dataset.flux.data


class AbstractNear2FarMonitorData(MonitorData, ABC):
    """Collection of radiation vectors in the frequency domain."""

    monitor: Union[Near2FarAngleMonitor, Near2FarCartesianMonitor, Near2FarKSpaceMonitor]
    dataset: Union[Near2FarAngleData, Near2FarCartesianData, Near2FarKSpaceData]

    @property
    def Ltheta(self) -> xr.DataArray:
        """theta-component of the 'L' radiation vector."""
        return self.dataset.Ltheta.data

    @property
    def Lphi(self) -> xr.DataArray:
        """phi-component of the 'L' radiation vector."""
        return self.dataset.Lphi.data

    @property
    def Ntheta(self) -> xr.DataArray:
        """theta-component of the 'N' radiation vector."""
        return self.dataset.Ntheta.data

    @property
    def Nphi(self) -> xr.DataArray:
        """phi-component of the 'N' radiation vector."""
        return self.dataset.Nphi.data

    def normalize(
        self, source_spectrum_fn: Callable[[float], complex]
    ) -> AbstractNear2FarMonitorData:
        """Return copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        for field_name, field_data in self.dataset.field_components.items():
            src_amps = source_spectrum_fn(field_data.data.f)
            new_data_array = field_data.data / src_amps
            fields_norm[field_name] = field_data.copy(update=dict(data=new_data_array))
        new_dataset = self.dataset.copy(update=fields_norm)
        return self.copy(update=dict(dataset=new_dataset))

    @staticmethod
    def nk(frequency: float, medium: Medium) -> Tuple[float, float]:
        """Returns the real and imaginary parts of the background medium's refractive index."""
        eps_complex = medium.eps_model(frequency)
        return medium.eps_complex_to_nk(eps_complex)

    @staticmethod
    def k(frequency: float, medium: Medium) -> complex:
        """Returns the complex wave number associated with the background medium."""
        index_n, index_k = AbstractNear2FarMonitorData.nk(frequency, medium)
        return (2 * np.pi * frequency / C_0) * (index_n + 1j * index_k)

    @staticmethod
    def eta(frequency: float, medium: Medium) -> complex:
        """Returns the complex wave impedance associated with the background medium."""
        eps_complex = medium.eps_model(frequency)
        return ETA_0 / np.sqrt(eps_complex)

    @staticmethod
    def car_2_sph(x: float, y: float, z: float) -> Tuple[float, float, float]:
        """Convert Cartesian to spherical coordinates.

        Parameters
        ----------
        x : float
            x coordinate relative to ``local_origin``.
        y : float
            y coordinate relative to ``local_origin``.
        z : float
            z coordinate relative to ``local_origin``.

        Returns
        -------
        Tuple[float, float, float]
            r, theta, and phi coordinates relative to ``local_origin``.
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi

    @staticmethod
    def sph_2_car(r, theta, phi) -> Tuple[float, float, float]:
        """Convert spherical to Cartesian coordinates.

        Parameters
        ----------
        r : float
            radius.
        theta : float
            polar angle (rad) downward from x=y=0 line.
        phi : float
            azimuthal (rad) angle from y=z=0 line.

        Returns
        -------
        Tuple[float, float, float]
            x, y, and z coordinates relative to ``local_origin``.
        """
        r_sin_theta = r * np.sin(theta)
        x = r_sin_theta * np.cos(phi)
        y = r_sin_theta * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    @staticmethod
    def sph_2_car_field(f_r, f_theta, f_phi, theta, phi) -> Tuple[complex, complex, complex]:
        """Convert vector field components in spherical coordinates to cartesian.

        Parameters
        ----------
        f_r : float
            radial component of the vector field.
        f_theta : float
            polar angle component of the vector fielf.
        f_phi : float
            azimuthal angle component of the vector field.
        theta : float
            polar angle (rad) of location of the vector field.
        phi : float
            azimuthal angle (rad) of location of the vector field.

        Returns
        -------
        Tuple[float, float, float]
            x, y, and z components of the vector field in cartesian coordinates.
        """
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        f_x = f_r * sin_theta * cos_phi + f_theta * cos_theta * cos_phi - f_phi * sin_phi
        f_y = f_r * sin_theta * sin_phi + f_theta * cos_theta * sin_phi + f_phi * cos_phi
        f_z = f_r * cos_theta - f_theta * sin_theta
        return f_x, f_y, f_z

    @staticmethod
    def kspace_2_sph(ux, uy, axis) -> Tuple[float, float]:
        """Convert normalized k-space coordinates to angles.

        Parameters
        ----------
        ux : float
            normalized kx coordinate.
        uy : float
            normalized ky coordinate.
        axis : int
            axis along which the observation plane is oriented.

        Returns
        -------
        Tuple[float, float]
            theta and phi coordinates relative to ``local_origin``.
        """
        phi_local = np.arctan2(uy, ux)
        theta_local = np.arcsin(np.sqrt(ux**2 + uy**2))
        # Spherical coordinates rotation matrix reference:
        # https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula#Matrix_notation
        if axis == 0:
            x = np.cos(theta_local)
            y = np.sin(theta_local) * np.sin(phi_local)
            z = -np.sin(theta_local) * np.cos(phi_local)
            theta = np.arccos(z)
            phi = np.arctan2(y, x)
        elif axis == 1:
            x = np.sin(theta_local) * np.cos(phi_local)
            y = np.cos(theta_local)
            z = -np.sin(theta_local) * np.sin(phi_local)
            theta = np.arccos(z)
            phi = np.arctan2(y, x)
        elif axis == 2:
            theta = theta_local
            phi = phi_local
        return theta, phi


class Near2FarAngleMonitorData(AbstractNear2FarMonitorData):
    """Data associated with a :class:`.Near2FarAngleMonitor`: components of radiation vectors.

    Example
    -------
    >>> from .data_array import Near2FarAngleDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(theta=theta, phi=phi, f=f)
    >>> values = (1+1j) * np.random.random((len(theta), len(phi), len(f)))
    >>> data_array = xr.DataArray(values, coords=coords)
    >>> scalar_field = Near2FarAngleDataArray(data=data_array)
    >>> dataset = Near2FarAngleData(
    ...     Ntheta=scalar_field,
    ...     Nphi=scalar_field,
    ...     Ltheta=scalar_field,
    ...     Lphi=scalar_field
    ... )
    >>> monitor = Near2FarAngleMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', phi=phi, theta=theta
    ... )
    >>> data = Near2FarAngleMonitorData(monitor=monitor, dataset=dataset)
    """

    monitor: Near2FarAngleMonitor
    dataset: Near2FarAngleData

    _contains_monitor_fields = enforce_monitor_fields_present()

    # pylint:disable=too-many-locals
    def fields(self, r: float = None, medium: Medium = Medium(permittivity=1)) -> xr.Dataset:
        """Get fields in spherical coordinates relative to the monitor's local origin
        for all angles and frequencies specified in :class:`Near2FarAngleMonitor`.
        If the radial distance ``r`` is provided, a corresponding phase factor is applied
        to the returned fields.

        Parameters
        ----------
        r : float = None
            (micron) radial distance relative to the monitor's local origin.
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: free space.

        Returns
        -------
        ``xarray.Dataset``
            xarray dataset containing (Er, Etheta, Ephi), (Hr, Htheta, Hphi)
            in polar coordinates.
        """
        return self.dataset.fields(r=r, medium=medium)

    def radar_cross_section(self, medium: Medium = Medium(permittivity=1)) -> xr.DataArray:
        """Get radar cross section at the observation grid in units of incident power.

        Parameters
        ----------
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: free space.

        Returns
        -------
        ``xarray.DataArray``
            Radar cross section at angles relative to the local origin.
        """
        return self.dataset.radar_cross_section(medium=medium)

    def power(self, r: float, medium: Medium = Medium(permittivity=1)) -> xr.DataArray:
        """Get power measured on the observation grid defined in spherical coordinates.

        Parameters
        ----------
        r : float
            (micron) radial distance relative to the local origin.
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: free space.

        Returns
        -------
        ``xarray.DataArray``
            Power at points relative to the local origin.
        """

        return self.dataset.power(r=r, medium=medium)


class Near2FarCartesianMonitorData(AbstractNear2FarMonitorData):
    """Data associated with a :class:`.Near2FarCartesianMonitor`: components of radiation vectors.

    Example
    -------
    >>> from .data_array import Near2FarCartesianDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> x = np.linspace(0, 5, 10)
    >>> y = np.linspace(0, 10, 20)
    >>> coords = dict(x=x, y=y, f=f)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(f)))
    >>> data_array = xr.DataArray(values, coords=coords)
    >>> scalar_field = Near2FarCartesianDataArray(data=data_array)
    >>> dataset = Near2FarCartesianData(
    ...     Ntheta=scalar_field,
    ...     Nphi=scalar_field,
    ...     Ltheta=scalar_field,
    ...     Lphi=scalar_field
    ... )
    >>> monitor = Near2FarCartesianMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', x=x, y=y,
    ...     plane_axis=2, plane_distance=50
    ... )
    >>> data = Near2FarCartesianMonitorData(monitor=monitor, dataset=dataset)
    """

    monitor: Near2FarCartesianMonitor
    dataset: Near2FarCartesianData

    # pylint:disable=too-many-arguments, too-many-locals
    def fields(self, medium: Medium = Medium(permittivity=1)) -> xr.Dataset:
        """Get fields on a cartesian plane at a distance relative to monitor center
        along a given axis.

        Parameters
        ----------
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: free space.

        Returns
        -------
        ``xarray.Dataset``
            xarray dataset containing (Ex, Ey, Ez), (Hx, Hy, Hz) in cartesian coordinates.
        """
        return self.dataset.fields(
            medium=medium,
            plane_distance=self.monitor.plane_distance,
            plane_axis=self.monitor.plane_axis,
        )

    def power(self, medium: Medium = Medium(permittivity=1)) -> xr.Dataset:
        """Get power on the observation grid defined in Cartesian coordinates.

        Parameters
        ----------
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: free space.

        Returns
        -------
        ``xarray.DataArray``
            Power at points relative to the local origin.
        """
        return self.dataset.power(
            medium=medium,
            plane_distance=self.monitor.plane_distance,
            plane_axis=self.monitor.plane_axis,
        )


class Near2FarKSpaceMonitorData(AbstractNear2FarMonitorData):
    """Data associated with a :class:`.Near2FarKSpaceMonitor`: components of radiation vectors.

    Example
    -------
    >>> from .data_array import Near2FarKSpaceDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> ux = np.linspace(0, 5, 10)
    >>> uy = np.linspace(0, 10, 20)
    >>> coords = dict(ux=ux, uy=uy, f=f)
    >>> values = (1+1j) * np.random.random((len(ux), len(uy), len(f)))
    >>> data_array = xr.DataArray(values, coords=coords)
    >>> scalar_field = Near2FarKSpaceDataArray(data=data_array)
    >>> dataset = Near2FarKSpaceData(
    ...     Ntheta=scalar_field,
    ...     Nphi=scalar_field,
    ...     Ltheta=scalar_field,
    ...     Lphi=scalar_field
    ... )
    >>> monitor = Near2FarKSpaceMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', ux=ux, uy=uy, u_axis=2
    ... )
    >>> data = Near2FarKSpaceMonitorData(monitor=monitor, dataset=dataset)
    """

    monitor: Near2FarKSpaceMonitor
    dataset: Near2FarKSpaceData

    # pylint:disable=too-many-locals
    def fields(self, medium: Medium = Medium(permittivity=1)) -> xr.Dataset:
        """Get fields in spherical coordinates relative to the monitor's local origin
        for all k-space points and frequencies specified in :class:`Near2FarKSpaceMonitor`.

        Parameters
        ----------
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: free space.

        Returns
        -------
        ``xarray.Dataset``
            xarray dataset containing (Er, Etheta, Ephi), (Hr, Htheta, Hphi)
            in polar coordinates.
        """

        return self.dataset.fields(medium=medium)

    def power(self, medium: Medium = Medium(permittivity=1)) -> xr.Dataset:
        """Get power on the observation grid defined in k-space.

        Parameters
        ----------
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: free space.

        Returns
        -------
        ``xarray.Dataset``
            xarray dataset containing power.
        """

        return self.dataset.power(medium=medium)


MonitorDataTypes = (
    FieldMonitorData,
    FieldTimeMonitorData,
    PermittivityMonitorData,
    ModeSolverMonitorData,
    ModeMonitorData,
    FluxMonitorData,
    FluxTimeMonitorData,
    Near2FarAngleMonitorData,
    Near2FarCartesianMonitorData,
    Near2FarKSpaceMonitorData,
)

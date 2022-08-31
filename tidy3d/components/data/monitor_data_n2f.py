""" Monitor Level Data associated with near-to-far transformation monitors."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Union, Callable, Tuple

import xarray as xr
import numpy as np
import pydantic as pd

from .monitor_data import MonitorData
from .data_array import DataArray
from .data_array import Near2FarAngleDataArray, Near2FarCartesianDataArray, Near2FarKSpaceDataArray
from ..monitor import Near2FarAngleMonitor, Near2FarCartesianMonitor, Near2FarKSpaceMonitor
from ..medium import Medium
from ..validators import enforce_monitor_fields_present
from ...log import SetupError, log
from ...constants import C_0, ETA_0


RADVECTYPE = Union[Near2FarAngleDataArray, Near2FarCartesianDataArray, Near2FarKSpaceDataArray]


class AbstractNear2FarData(MonitorData, ABC):
    """Collection of radiation vectors in the frequency domain."""

    monitor: Union[Near2FarAngleMonitor, Near2FarCartesianMonitor, Near2FarKSpaceMonitor] = None

    Ntheta: RADVECTYPE = pd.Field(
        ...,
        title="Ntheta",
        description="Spatial distribution of the theta-component of the N radiation vector.",
    )
    Nphi: RADVECTYPE = pd.Field(
        ...,
        title="Nphi",
        description="Spatial distribution of phi-component of the N radiation vector.",
    )
    Ltheta: RADVECTYPE = pd.Field(
        ...,
        title="Ltheta",
        description="Spatial distribution of theta-component of the L radiation vector.",
    )
    Lphi: RADVECTYPE = pd.Field(
        ...,
        title="Lphi",
        description="Spatial distribution of phi-component of the L radiation vector.",
    )

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to thier associated data."""
        return dict(
            Ntheta=self.Ntheta,
            Nphi=self.Nphi,
            Ltheta=self.Ltheta,
            Lphi=self.Lphi,
        )

    @property
    def f(self) -> np.ndarray:
        """Frequencies."""
        return self.Ntheta.f.values

    @property
    def coords(self) -> Dict[str, np.ndarray]:
        """Coordinates of the radiation vectors contained."""
        return self.Ntheta.coords

    @property
    def dims(self) -> Tuple[str, ...]:
        """Dimensions of the radiation vectors contained."""
        return self.Ntheta.__slots__

    def make_data_array(self, data: np.ndarray) -> xr.DataArray:
        """Make an xr.DataArray with data and same coords and dims as radiation vectors of self."""
        return xr.DataArray(data=data, coords=self.coords, dims=self.dims)

    def make_dataset(self, keys: Tuple[str, ...], vals: Tuple[np.ndarray, ...]) -> xr.Dataset:
        """Make an xr.Dataset with keys and data with same coords and dims as radiation vectors."""
        data_arrays = tuple(map(self.make_data_array, vals))
        return xr.Dataset(dict(zip(keys, data_arrays)))

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> AbstractNear2FarData:
        """Return copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        for field_name, field_data in self.field_components.items():
            src_amps = source_spectrum_fn(field_data.f)
            fields_norm[field_name] = field_data / src_amps

        return self.copy(update=fields_norm)

    def nk(self, medium: Medium) -> Tuple[float, float]:
        """Returns the real and imaginary parts of the background medium's refractive index."""
        return medium.nk_model(frequency=self.f)

    @staticmethod
    def propagation_factor(frequency: float, medium: Medium) -> complex:
        """Complex valued wavenumber associated with a frequency and medium."""
        index_n, index_k = medium.nk_model(frequency=frequency)
        return (2 * np.pi * frequency / C_0) * (index_n + 1j * index_k)

    def k(self, medium: Medium) -> complex:
        """Returns the complex wave number associated with the background medium."""
        return self.propagation_factor(frequency=self.f, medium=medium)

    def eta(self, medium: Medium) -> complex:
        """Returns the complex wave impedance associated with the background medium."""
        eps_complex = medium.eps_model(frequency=self.f)
        return ETA_0 / np.sqrt(eps_complex)

    def rad_vecs_to_fields(self, medium: Medium) -> Tuple[np.ndarray, np.ndarray]:
        """Compute fields from radiation vectors."""
        eta = self.eta(medium=medium)
        e_theta = -(self.Lphi.values + eta * self.Ntheta.values)
        e_phi = self.Ltheta.values - eta * self.Nphi.values
        return e_theta, e_phi

    @staticmethod
    def propagation_phase(dist: Union[float, None], wave_number: complex) -> complex:
        """Phase associated with propagation of a distance with a given wavenumber."""
        if dist is None:
            return 1.0
        return -1j * wave_number * np.exp(1j * wave_number * dist) / (4 * np.pi * dist)

    def fields_sph(self, r: float = None, medium: Medium = Medium(permittivity=1)) -> xr.Dataset:
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
            xarray dataset containing
            (``E_r``, ``E_theta``, ``E_phi``, ``H_r``, ``H_theta``, ``H_phi``)
            in polar coordinates.
        """

        # assemble E felds
        e_theta, e_phi = self.rad_vecs_to_fields(medium=medium)
        phase = self.propagation_phase(dist=r, wave_number=self.k(medium=medium))
        Et_array = phase * e_theta
        Ep_array = phase * e_phi
        Er_array = np.zeros_like(Ep_array)

        # assemble H fields
        eta = self.eta(medium=medium)[None, None, :]
        Ht_array = -Ep_array / eta
        Hp_array = Et_array / eta
        Hr_array = np.zeros_like(Hp_array)

        keys = ("E_r", "E_theta", "E_phi", "H_r", "H_theta", "H_phi")
        vals = (Er_array, Et_array, Ep_array, Hr_array, Ht_array, Hp_array)
        return self.make_dataset(keys=keys, vals=vals)

    @abstractmethod
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
            xarray dataset containing
            (``E_r``, ``E_theta``, ``E_phi``, ``H_r``, ``H_theta``, ``H_phi``)
            in polar coordinates.
        """

    @abstractmethod
    def power(self, r: float = None, medium: Medium = Medium(permittivity=1)) -> xr.DataArray:
        """Get power measured on the observation grid defined in spherical coordinates.

        Parameters
        ----------
        r : float = None
            (micron) radial distance relative to the local origin.
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: free space.

        Returns
        -------
        ``xarray.DataArray``
            Power at points relative to the local origin.
        """


class Near2FarAngleData(AbstractNear2FarData):
    """Data associated with a :class:`.Near2FarAngleMonitor`: components of radiation vectors.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(f=f, theta=theta, phi=phi)
    >>> values = (1+1j) * np.random.random((len(theta), len(phi), len(f)))
    >>> scalar_field = Near2FarAngleDataArray(values, coords=coords)
    >>> monitor = Near2FarAngleMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', phi=phi, theta=theta
    ...     )
    >>> data = Near2FarAngleData(
    ...     monitor=monitor, Ntheta=scalar_field, Nphi=scalar_field,
    ...     Ltheta=scalar_field, Lphi=scalar_field
    ...     )
    """

    monitor: Near2FarAngleMonitor = None

    Ntheta: Near2FarAngleDataArray = pd.Field(
        ...,
        title="Ntheta",
        description="Spatial distribution of the theta-component of the N radiation vector.",
    )
    Nphi: Near2FarAngleDataArray = pd.Field(
        ...,
        title="Nphi",
        description="Spatial distribution of phi-component of the N radiation vector.",
    )
    Ltheta: Near2FarAngleDataArray = pd.Field(
        ...,
        title="Ltheta",
        description="Spatial distribution of theta-component of the L radiation vector.",
    )
    Lphi: Near2FarAngleDataArray = pd.Field(
        ...,
        title="Lphi",
        description="Spatial distribution of phi-component of the L radiation vector.",
    )

    _contains_monitor_fields = enforce_monitor_fields_present()

    @property
    def theta(self) -> np.ndarray:
        """Polar angles."""
        return self.Ntheta.theta.values

    @property
    def phi(self) -> np.ndarray:
        """Azimuthal angles."""
        return self.Ntheta.phi.values

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
            xarray dataset containing
            (``E_r``, ``E_theta``, ``E_phi``, ``H_r``, ``H_theta``, ``H_phi``)
            in polar coordinates.
        """
        return self.fields_sph(r=r, medium=medium)

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

        _, index_k = self.nk(medium=medium)
        if not np.all(index_k == 0):
            raise SetupError("Can't compute RCS for a lossy background medium.")

        k = self.k(medium=medium)[None, None, ...]
        eta = self.eta(medium=medium)[None, None, ...]

        constant = k**2 / (8 * np.pi * eta)
        e_theta, e_phi = self.rad_vecs_to_fields(medium=medium)
        rcs_data = constant * (np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2)

        return self.make_data_array(data=rcs_data)

    def power(self, r: float = None, medium: Medium = Medium(permittivity=1)) -> xr.DataArray:
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

        if r is None:
            raise ValueError("'r' required by 'Near2FarAngleData.power'")

        field_data = self.fields(medium=medium, r=r)
        Et, Ep = [field_data[comp].values for comp in ["E_theta", "E_phi"]]
        Ht, Hp = [field_data[comp].values for comp in ["H_theta", "H_phi"]]
        power_theta = 0.5 * np.real(Et * np.conj(Hp))
        power_phi = 0.5 * np.real(-Ep * np.conj(Ht))
        power_data = power_theta + power_phi

        return self.make_data_array(data=power_data)


class Near2FarCartesianData(AbstractNear2FarData):
    """Data associated with a :class:`.Near2FarCartesianMonitor`: components of radiation vectors.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> x = np.linspace(0, 5, 10)
    >>> y = np.linspace(0, 10, 20)
    >>> coords = dict(f=f, x=x, y=y)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(f)))
    >>> scalar_field = Near2FarCartesianDataArray(values, coords=coords)
    >>> monitor = Near2FarCartesianMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', x=x, y=y,
    ...     plane_axis=2, plane_distance=50
    ...     )
    >>> data = Near2FarCartesianData(
    ...     monitor=monitor, Ntheta=scalar_field, Nphi=scalar_field,
    ...     Ltheta=scalar_field, Lphi=scalar_field
    ...     )
    """

    monitor: Near2FarCartesianMonitor

    Ntheta: Near2FarCartesianDataArray = pd.Field(
        ...,
        title="Ntheta",
        description="Spatial distribution of the theta-component of the N radiation vector.",
    )
    Nphi: Near2FarCartesianDataArray = pd.Field(
        ...,
        title="Nphi",
        description="Spatial distribution of the phi-component of the N radiation vector.",
    )
    Ltheta: Near2FarCartesianDataArray = pd.Field(
        ...,
        title="Ltheta",
        description="Spatial distribution of the theta-component of the L radiation vector.",
    )
    Lphi: Near2FarCartesianDataArray = pd.Field(
        ...,
        title="Lphi",
        description="Spatial distribution of the phi-component of the L radiation vector.",
    )

    _contains_monitor_fields = enforce_monitor_fields_present()

    @property
    def x(self) -> np.ndarray:
        """X positions."""
        return self.Ntheta.x.values

    @property
    def y(self) -> np.ndarray:
        """Y positions."""
        return self.Ntheta.y.values

    @property
    def spherical_coords(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The data coordinates in spherical coordinate system."""
        xs, ys, _ = np.meshgrid(self.x, self.y, np.array([0]), indexing="ij")
        zs = self.monitor.plane_distance * np.ones_like(xs)
        coords = [xs, ys]
        coords.insert(self.monitor.plane_axis, zs)
        x_glob, y_glob, z_glob = coords
        return self.monitor.car_2_sph(x_glob, y_glob, z_glob)

    # pylint:disable=too-many-arguments, too-many-locals
    def fields(self, r: float = None, medium: Medium = Medium(permittivity=1)) -> xr.Dataset:
        """Get fields on a cartesian plane at a distance relative to monitor center
        along a given axis in cartesian coordinates.

        Parameters
        ----------
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: free space.

        Returns
        -------
        ``xarray.Dataset``
            xarray dataset containing (``Ex``, ``Ey``, ``Ez``, ``Hx``, ``Hy``, ``Hz``)
            in cartesian coordinates.
        """

        if r is not None:
            log.warning(
                "'r' supplied to 'Near2FarCartesianData.fields' will not be used. "
                "distance is determined from the coordinates contained in the data."
            )

        # get the fields in spherical coordinates
        r_values, thetas, phis = self.spherical_coords
        fields_sph = self.fields_sph(r=r_values, medium=medium)
        Er, Et, Ep = (fields_sph[key].values for key in ("E_r", "E_theta", "E_phi"))
        Hr, Ht, Hp = (fields_sph[key].values for key in ("H_r", "H_theta", "H_phi"))

        # convert the field components to cartesian coordinate system
        e_data = self.monitor.sph_2_car_field(Er, Et, Ep, thetas, phis)
        h_data = self.monitor.sph_2_car_field(Hr, Ht, Hp, thetas, phis)

        # package into dataset
        keys = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        field_components = np.concatenate((e_data, h_data), axis=0)
        return self.make_dataset(keys=keys, vals=field_components)

    def power(self, r: float = None, medium: Medium = Medium(permittivity=1)) -> xr.Dataset:
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

        if r is not None:
            log.warning(
                "'r' supplied to 'Near2FarCartesianData.power' will not be used. "
                "distance is determined from the coordinates contained in the data."
            )

        # get the polar components of the far field
        r_values, _, _ = self.spherical_coords
        fields_sph = self.fields_sph(r=r_values, medium=medium)
        Et, Ep = (fields_sph[key].values for key in ("E_theta", "E_phi"))
        Ht, Hp = (fields_sph[key].values for key in ("H_theta", "H_phi"))

        # compute power
        power_theta = 0.5 * np.real(Et * np.conj(Hp))
        power_phi = 0.5 * np.real(-Ep * np.conj(Ht))
        power = power_theta + power_phi

        # package as data array
        return self.make_data_array(data=power)


class Near2FarKSpaceData(AbstractNear2FarData):
    """Data associated with a :class:`.Near2FarKSpaceMonitor`: components of radiation vectors.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> ux = np.linspace(0, 5, 10)
    >>> uy = np.linspace(0, 10, 20)
    >>> coords = dict(f=f, ux=ux, uy=uy)
    >>> values = (1+1j) * np.random.random((len(ux), len(uy), len(f)))
    >>> scalar_field = Near2FarKSpaceDataArray(values, coords=coords)
    >>> monitor = Near2FarKSpaceMonitor(
    ...     center=(1,2,3), size=(2,2,2), freqs=f, name='n2f_monitor', ux=ux, uy=uy, u_axis=2
    ...     )
    >>> data = Near2FarKSpaceData(
    ...     monitor=monitor, Ntheta=scalar_field, Nphi=scalar_field,
    ...     Ltheta=scalar_field, Lphi=scalar_field
    ...     )
    """

    monitor: Near2FarKSpaceMonitor = None

    Ntheta: Near2FarKSpaceDataArray = pd.Field(
        ...,
        title="Ntheta",
        description="Spatial distribution of the theta-component of the N radiation vector.",
    )
    Nphi: Near2FarKSpaceDataArray = pd.Field(
        ...,
        title="Nphi",
        description="Spatial distribution of phi-component of the N radiation vector.",
    )
    Ltheta: Near2FarKSpaceDataArray = pd.Field(
        ...,
        title="Ltheta",
        description="Spatial distribution of theta-component of the L radiation vector.",
    )
    Lphi: Near2FarKSpaceDataArray = pd.Field(
        ...,
        title="Lphi",
        description="Spatial distribution of phi-component of the L radiation vector.",
    )

    _contains_monitor_fields = enforce_monitor_fields_present()

    @property
    def ux(self) -> np.ndarray:
        """reciprocal X positions."""
        return self.Ntheta.ux.values

    @property
    def uy(self) -> np.ndarray:
        """reciprocal Y positions."""
        return self.Ntheta.uy.values

    # pylint:disable=too-many-locals
    def fields(self, r: float = None, medium: Medium = Medium(permittivity=1)) -> xr.Dataset:
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
            xarray dataset containing
            (``E_r``, ``E_theta``, ``E_phi``, ``H_r``, ``H_theta``, ``H_phi``)
            in polar coordinates.
        """

        if r is not None:
            log.warning("'r' supplied to 'Near2FarKSpaceData.fields' will not be used.")

        # assemble E felds
        eta = self.eta(medium=medium)[None, None, ...]
        Et_array = -(self.Lphi.values + eta * self.Ntheta.values)
        Ep_array = self.Ltheta.values - eta * self.Nphi.values
        Er_array = np.zeros_like(Ep_array)

        # assemble H fields
        Ht_array = -Ep_array / eta
        Hp_array = Et_array / eta
        Hr_array = np.zeros_like(Hp_array)

        keys = ("E_r", "E_theta", "E_phi", "H_r", "H_theta", "H_phi")
        vals = (Er_array, Et_array, Ep_array, Hr_array, Ht_array, Hp_array)
        return self.make_dataset(keys=keys, vals=vals)

    def power(self, r: float = None, medium: Medium = Medium(permittivity=1)) -> xr.Dataset:
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

        if r is not None:
            log.warning("'r' supplied to 'Near2FarKSpaceData.fields' will not be used.")

        fields = self.fields(r=None, medium=medium)
        Et, Ep, Ht, Hp = (fields[key].values for key in ("E_theta", "E_phi", "H_theta", "H_phi"))

        power_theta = 0.5 * np.real(Et * np.conj(Hp))
        power_phi = 0.5 * np.real(-Ep * np.conj(Ht))
        power_values = power_theta + power_phi

        return self.make_data_array(data=power_values)


Near2FarDataTypes = (Near2FarAngleData, Near2FarCartesianData, Near2FarKSpaceData)

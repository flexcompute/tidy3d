""" Monitor Level Data associated with near-to-far transformation monitors."""
from __future__ import annotations
from typing import Dict, Union, Callable, Tuple

import xarray as xr
import numpy as np
import pydantic as pd

from ..monitor import Near2FarAngleMonitor, Near2FarCartesianMonitor, Near2FarKSpaceMonitor
from ..medium import Medium
from ..validators import enforce_monitor_fields_present
from ...log import SetupError
from ...constants import C_0, ETA_0

from .monitor_data import MonitorData
from .data_array import DataArray
from .data_array import Near2FarAngleDataArray, Near2FarCartesianDataArray, Near2FarKSpaceDataArray


class AbstractNear2FarData(MonitorData):
    """Collection of radiation vectors in the frequency domain."""

    monitor: Union[Near2FarAngleMonitor, Near2FarCartesianMonitor, Near2FarKSpaceMonitor]

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to thier associated data."""
        return {field: getattr(self, field) for field in self.monitor.fields}

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> AbstractNear2FarData:
        """Return copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        for field_name, field_data in self.field_components.items():
            src_amps = source_spectrum_fn(field_data.f)
            fields_norm[field_name] = field_data / src_amps

        return self.copy(update=fields_norm)

    @staticmethod
    def nk(frequency: float, medium: Medium) -> Tuple[float, float]:
        """Returns the real and imaginary parts of the background medium's refractive index."""
        eps_complex = medium.eps_model(frequency)
        return medium.eps_complex_to_nk(eps_complex)

    @staticmethod
    def k(frequency: float, medium: Medium) -> complex:
        """Returns the complex wave number associated with the background medium."""
        index_n, index_k = AbstractNear2FarData.nk(frequency, medium)
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

    monitor: Near2FarAngleMonitor

    Ntheta: Near2FarAngleDataArray = pd.Field(
        None,
        title="Ntheta",
        description="Spatial distribution of the theta-component of the N radiation vector.",
    )
    Nphi: Near2FarAngleDataArray = pd.Field(
        None,
        title="Nphi",
        description="Spatial distribution of phi-component of the N radiation vector.",
    )
    Ltheta: Near2FarAngleDataArray = pd.Field(
        None,
        title="Ltheta",
        description="Spatial distribution of theta-component of the L radiation vector.",
    )
    Lphi: Near2FarAngleDataArray = pd.Field(
        None,
        title="Lphi",
        description="Spatial distribution of phi-component of the L radiation vector.",
    )

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

        frequencies = np.array(self.monitor.freqs)
        theta = np.array(self.monitor.theta)
        phi = np.array(self.monitor.phi)

        k = np.array([self.k(frequency, medium) for frequency in frequencies])
        eta = np.array([self.eta(frequency, medium) for frequency in frequencies])
        eta = eta[None, None, :]

        # assemble E felds
        if r is not None:
            phase = -1j * k * np.exp(1j * k * r) / (4 * np.pi * r)
            phase = phase[None, None, :]
        else:
            phase = 1.0

        Et_array = -phase * (self.Lphi.values + eta * self.Ntheta.values)
        Ep_array = phase * (self.Ltheta.values - eta * self.Nphi.values)
        Er_array = np.zeros_like(Ep_array)

        dims = ("theta", "phi", "f")
        coords = {"theta": theta, "phi": phi, "f": frequencies}

        # assemble H fields
        Ht_array = -Ep_array / eta
        Hp_array = Et_array / eta
        Hr_array = np.zeros_like(Hp_array)

        Er = xr.DataArray(data=Er_array, coords=coords, dims=dims)
        Et = xr.DataArray(data=Et_array, coords=coords, dims=dims)
        Ep = xr.DataArray(data=Ep_array, coords=coords, dims=dims)

        Hr = xr.DataArray(data=Hr_array, coords=coords, dims=dims)
        Ht = xr.DataArray(data=Ht_array, coords=coords, dims=dims)
        Hp = xr.DataArray(data=Hp_array, coords=coords, dims=dims)

        field_data = xr.Dataset(
            {"E_r": Er, "E_theta": Et, "E_phi": Ep, "H_r": Hr, "H_theta": Ht, "H_phi": Hp}
        )

        return field_data

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

        frequencies = np.array(self.monitor.freqs)
        theta = np.array(self.monitor.theta)
        phi = np.array(self.monitor.phi)

        for frequency in frequencies:
            _, index_k = self.nk(frequency, medium)
            if index_k != 0.0:
                raise SetupError("Can't compute RCS for a lossy background medium.")

        k = np.array([self.k(frequency, medium) for frequency in frequencies])
        eta = np.array([self.eta(frequency, medium) for frequency in frequencies])

        k = k[None, None, ...]
        eta = eta[None, None, ...]

        constant = k**2 / (8 * np.pi * eta)
        term1 = np.abs(self.Lphi.values + eta * self.Ntheta.values) ** 2
        term2 = np.abs(self.Ltheta.values - eta * self.Nphi.values) ** 2
        rcs_data = constant * (term1 + term2)

        dims = ("theta", "phi", "f")
        coords = {"theta": theta, "phi": phi, "f": frequencies}

        return xr.DataArray(data=rcs_data, coords=coords, dims=dims)

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

        field_data = self.fields(medium=medium, r=r)
        Et, Ep = [field_data[comp].values for comp in ["E_theta", "E_phi"]]
        Ht, Hp = [field_data[comp].values for comp in ["H_theta", "H_phi"]]
        power_theta = 0.5 * np.real(Et * np.conj(Hp))
        power_phi = 0.5 * np.real(-Ep * np.conj(Ht))
        power_data = power_theta + power_phi

        dims = ("theta", "phi", "f")
        coords = {
            "theta": np.array(self.monitor.theta),
            "phi": np.array(self.monitor.phi),
            "f": np.array(self.monitor.freqs),
        }

        return xr.DataArray(data=power_data, coords=coords, dims=dims)


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
        None,
        title="Ntheta",
        description="Spatial distribution of the theta-component of the N radiation vector.",
    )
    Nphi: Near2FarCartesianDataArray = pd.Field(
        None,
        title="Nphi",
        description="Spatial distribution of the phi-component of the N radiation vector.",
    )
    Ltheta: Near2FarCartesianDataArray = pd.Field(
        None,
        title="Ltheta",
        description="Spatial distribution of the theta-component of the L radiation vector.",
    )
    Lphi: Near2FarCartesianDataArray = pd.Field(
        None,
        title="Lphi",
        description="Spatial distribution of the phi-component of the L radiation vector.",
    )

    _contains_monitor_fields = enforce_monitor_fields_present()

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

        frequencies = np.array(self.monitor.freqs)

        eta = np.atleast_1d(self.eta(frequencies, medium))
        wave_number = np.atleast_1d(self.k(frequencies, medium))

        e_theta = -(self.Lphi.values + eta[None, None, ...] * self.Ntheta.values)
        e_phi = self.Ltheta.values - eta[None, None, ...] * self.Nphi.values

        e_data = np.zeros(
            (3, len(self.monitor.x), len(self.monitor.y), len(frequencies)), dtype=complex
        )
        h_data = np.zeros_like(e_data)

        for i, _x in enumerate(self.monitor.x):
            for j, _y in enumerate(self.monitor.y):
                x_glob, y_glob, z_glob = self.monitor.unpop_axis(
                    self.monitor.plane_distance, (_x, _y), axis=self.monitor.plane_axis
                )
                r, theta, phi = self.car_2_sph(x_glob, y_glob, z_glob)
                phase = -1j * wave_number * np.exp(1j * wave_number * r) / (4 * np.pi * r)

                Et = -phase * e_theta[i, j, :]
                Ep = phase * e_phi[i, j, :]
                Er = np.zeros_like(Et)

                Ht = -Ep / eta
                Hp = Et / eta
                Hr = np.zeros_like(Hp)

                e_fields = self.sph_2_car_field(Er, Et, Ep, theta, phi)
                h_fields = self.sph_2_car_field(Hr, Ht, Hp, theta, phi)

                e_data[:, i, j, :] = e_fields
                h_data[:, i, j, :] = h_fields

        field_components = np.concatenate((e_data, h_data), axis=0)

        dims = ("x", "y", "f")
        coords = {"x": np.array(self.monitor.x), "y": np.array(self.monitor.y), "f": frequencies}

        fields = {
            key: xr.DataArray(data=val, coords=coords, dims=dims)
            for key, val in zip(("Ex", "Ey", "Ez", "Hx", "Hy", "Hz"), field_components)
        }
        return xr.Dataset(fields)

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

        frequencies = np.array(self.monitor.freqs)

        eta = np.atleast_1d(self.eta(frequencies, medium))
        wave_number = np.atleast_1d(self.k(frequencies, medium))

        e_theta = -(self.Lphi.values + eta[None, None, ...] * self.Ntheta.values)
        e_phi = self.Ltheta.values - eta[None, None, ...] * self.Nphi.values

        values = np.zeros((len(self.monitor.x), len(self.monitor.y), len(frequencies)))

        for i, _x in enumerate(self.monitor.x):
            for j, _y in enumerate(self.monitor.y):
                x_glob, y_glob, z_glob = self.monitor.unpop_axis(
                    self.monitor.plane_distance, (_x, _y), axis=self.monitor.plane_axis
                )
                r, _, _ = self.car_2_sph(x_glob, y_glob, z_glob)
                phase = -1j * wave_number * np.exp(1j * wave_number * r) / (4 * np.pi * r)

                Et = -phase * e_theta[i, j, :]
                Ep = phase * e_phi[i, j, :]

                Ht = -Ep / eta
                Hp = Et / eta

                power_theta = 0.5 * np.real(Et * np.conj(Hp))
                power_phi = 0.5 * np.real(-Ep * np.conj(Ht))
                values[i, j, :] = power_theta + power_phi

        dims = ("x", "y", "f")
        coords = {"x": np.array(self.monitor.x), "y": np.array(self.monitor.y), "f": frequencies}

        power_data = xr.DataArray(data=values, coords=coords, dims=dims)

        return power_data


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

    monitor: Near2FarKSpaceMonitor

    Ntheta: Near2FarKSpaceDataArray = pd.Field(
        None,
        title="Ntheta",
        description="Spatial distribution of the theta-component of the N radiation vector.",
    )
    Nphi: Near2FarKSpaceDataArray = pd.Field(
        None,
        title="Nphi",
        description="Spatial distribution of phi-component of the N radiation vector.",
    )
    Ltheta: Near2FarKSpaceDataArray = pd.Field(
        None,
        title="Ltheta",
        description="Spatial distribution of theta-component of the L radiation vector.",
    )
    Lphi: Near2FarKSpaceDataArray = pd.Field(
        None,
        title="Lphi",
        description="Spatial distribution of phi-component of the L radiation vector.",
    )

    _contains_monitor_fields = enforce_monitor_fields_present()

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

        # Assumes that frequencies and angles are the same for all radiation vectors
        frequencies = np.array(self.monitor.freqs)
        ux = np.array(self.monitor.ux)
        uy = np.array(self.monitor.uy)

        eta = np.array([self.eta(frequency, medium) for frequency in frequencies])

        # assemble E felds
        eta = eta[None, None, ...]
        Et_array = -(self.Lphi.values + eta * self.Ntheta.values)
        Ep_array = self.Ltheta.values - eta * self.Nphi.values
        Er_array = np.zeros_like(Ep_array)

        # assemble H fields
        Ht_array = -Ep_array / eta
        Hp_array = Et_array / eta
        Hr_array = np.zeros_like(Hp_array)

        dims = ("ux", "uy", "f")
        coords = {"ux": ux, "uy": uy, "f": frequencies}

        Er = xr.DataArray(data=Er_array, coords=coords, dims=dims)
        Et = xr.DataArray(data=Et_array, coords=coords, dims=dims)
        Ep = xr.DataArray(data=Ep_array, coords=coords, dims=dims)

        Hr = xr.DataArray(data=Hr_array, coords=coords, dims=dims)
        Ht = xr.DataArray(data=Ht_array, coords=coords, dims=dims)
        Hp = xr.DataArray(data=Hp_array, coords=coords, dims=dims)

        field_data = xr.Dataset(
            {"E_r": Er, "E_theta": Et, "E_phi": Ep, "H_r": Hr, "H_theta": Ht, "H_phi": Hp}
        )

        return field_data

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

        # Assumes that frequencies and angles are the same for all radiation vectors
        frequencies = np.array(self.monitor.freqs)
        ux = np.array(self.monitor.ux)
        uy = np.array(self.monitor.uy)

        eta = np.array([self.eta(frequency, medium) for frequency in frequencies])

        # assemble E felds
        eta = eta[None, None, ...]
        Et_array = -(self.Lphi.values + eta * self.Ntheta.values)
        Ep_array = self.Ltheta.values - eta * self.Nphi.values

        # assemble H fields
        Ht_array = -Ep_array / eta
        Hp_array = Et_array / eta

        power_theta = 0.5 * np.real(Et_array * np.conj(Hp_array))
        power_phi = 0.5 * np.real(-Ep_array * np.conj(Ht_array))
        values = power_theta + power_phi

        dims = ("ux", "uy", "f")
        coords = {"ux": ux, "uy": uy, "f": frequencies}

        power_data = xr.DataArray(data=values, coords=coords, dims=dims)

        return power_data


Near2FarDataTypes = (Near2FarAngleData, Near2FarCartesianData, Near2FarKSpaceData)

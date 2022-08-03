""" Monitor Level Data associated with near-to-far transformation monitors."""
from typing import Dict, Union, Callable, Tuple

import xarray as xr
import numpy as np
import pydantic as pd

from ..monitor import Near2FarAngleMonitor, Near2FarCartesianMonitor, Near2FarKSpaceMonitor
from ..medium import Medium
from ..validators import enforce_monitor_fields_present
from ...log import SetupError
from ...constants import C_0, ETA_0

from .monitor_data import MonitorData, DATA_TYPE_MAP
from .data_array import DataArray
from .data_array import Near2FarAngleDataArray, Near2FarCartesianDataArray, Near2FarKSpaceDataArray


class AbstractNear2FarData(MonitorData):
    """Collection of radiation vectors in the frequency domain."""

    monitor: Union[Near2FarAngleMonitor, Near2FarCartesianMonitor, Near2FarKSpaceMonitor]

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to thier associated data."""
        # pylint:disable=no-member
        return {field: getattr(self, field) for field in self.monitor.fields}

    @property
    def grid_locations(self) -> Dict[str, str]:
        """Maps field components to the string key of their locations on the observation grid."""
        return dict(Ntheta="Ntheta", Nphi="Nphi", Ltheta="Ltheta", Lphi="Lphi")

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> "AbstractNear2FarData":
        """Return copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        for field_name, field_data in self.field_components.items():
            src_amps = source_spectrum_fn(field_data.f)
            fields_norm[field_name] = field_data / src_amps

        return self.copy(update=fields_norm)

    def nk(self, frequency, medium) -> Tuple[float, float]:
        """Returns the real and imaginary parts of the background medium's refractive index."""
        eps_complex = medium.eps_model(frequency)
        return medium.eps_complex_to_nk(eps_complex)

    def k(self, frequency, medium) -> complex:
        """Returns the complex wave number associated with the background medium."""
        index_n, index_k = self.nk(frequency, medium)
        return (2 * np.pi * frequency / C_0) * (index_n + 1j * index_k)

    def eta(self, frequency, medium) -> complex:
        """Returns the complex wave impedance associated with the background medium."""
        eps_complex = medium.eps_model(frequency)
        return ETA_0 / np.sqrt(eps_complex)

    @staticmethod
    def car_2_sph(x: float, y: float, z: float):
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
        r : float
            r coordinate relative to ``local_origin``.
        theta : float
            theta coordinate relative to ``local_origin``.
        phi : float
            phi coordinate relative to ``local_origin``.
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return r, theta, phi

    @staticmethod
    def sph_2_car(r, theta, phi):
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
        x : float
            x coordinate relative to ``local_origin``.
        y : float
            y coordinate relative to ``local_origin``.
        z : float
            z coordinate relative to ``local_origin``.
        """
        r_sin_theta = r * np.sin(theta)
        x = r_sin_theta * np.cos(phi)
        y = r_sin_theta * np.sin(phi)
        z = r * np.cos(theta)
        return x, y, z

    @staticmethod
    def sph_2_car_field(f_r, f_theta, f_phi, theta, phi):
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
        tuple
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
    def fields_spherical(
        self, r: float = None, medium: Medium = Medium(permittivity=1)
    ) -> xr.Dataset:
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
        xarray.Dataset
            xarray dataset containing (Er, Etheta, Ephi), (Hr, Htheta, Hphi)
            in polar coordinates.
        """

        # Assumes that frequencies and angles are the same for all radiation vectors
        theta = self.Ntheta.theta
        phi = self.Ntheta.phi
        frequencies = self.Ntheta.f

        k = np.array([self.k(frequency, medium) for frequency in frequencies])
        eta = np.array([self.eta(frequency, medium) for frequency in frequencies])

        # assemble E felds
        if r is not None:
            phase = -1j * k * np.exp(1j * k * r) / (4 * np.pi * r)

            eta = eta[None, None, None, :]
            phase = phase[None, None, None, :]

            Et_array = -phase * (self.Lphi.values[None, ...] + eta * self.Ntheta.values[None, ...])
            Ep_array = phase * (self.Ltheta.values[None, ...] - eta * self.Nphi.values[None, ...])
            Er_array = np.zeros_like(Ep_array)

            dims = ("r", "theta", "phi", "f")
            coords = {"r": np.atleast_1d(r), "theta": theta, "phi": phi, "f": frequencies}

        else:
            eta = eta[None, None, ...]
            Et_array = -(self.Lphi.values + eta * self.Ntheta.values)
            Ep_array = self.Ltheta.values - eta * self.Nphi.values
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
        """Get radar cross section at a point relative to the local origin in
        units of incident power.

        Parameters
        ----------
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: free space.

        Returns
        -------
        RCS : xarray.DataArray
            Radar cross section at angles relative to the local origin.
        """

        # Assumes that frequencies and angles are the same for all radiation vectors
        frequencies = self.Ntheta.f
        theta = self.Ntheta.theta
        phi = self.Ntheta.phi

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

    def power_spherical(self, r: float, medium: Medium = Medium(permittivity=1)) -> xr.DataArray:
        """Get power scattered to a point relative to the local origin in spherical coordinates.

        Parameters
        ----------
        r : float
            (micron) radial distance relative to the local origin.

        Returns
        -------
        power : xarray.DataArray
            Power at points relative to the local origin.
        """

        field_data = self.fields_spherical(medium=medium, r=r)
        Et, Ep = [field_data[comp].values for comp in ["E_theta", "E_phi"]]
        Ht, Hp = [field_data[comp].values for comp in ["H_theta", "H_phi"]]
        power_theta = 0.5 * np.real(Et * np.conj(Hp))
        power_phi = 0.5 * np.real(-Ep * np.conj(Ht))
        power_data = power_theta + power_phi

        dims = ("r", "theta", "phi", "f")
        # Assumes that frequencies and angles are the same for all radiation vectors
        coords = {"r": [r], "theta": self.Ntheta.theta, "phi": self.Ntheta.phi, "f": self.Ntheta.f}

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
    def fields_cartesian(self, medium: Medium = Medium(permittivity=1)) -> xr.Dataset:
        """Get fields on a cartesian plane at a distance relative to monitor center
        along a given axis.

        Parameters
        ----------
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: free space.

        Returns
        -------
        xarray.Dataset
            xarray dataset containing (Ex, Ey, Ez), (Hx, Hy, Hz) in cartesian coordinates.
        """

        # Assumes that frequencies and coordinates are the same for all radiation vectors
        frequencies = self.Ntheta.f
        if self.monitor.plane_axis == 0:
            y = self.Ntheta.x
            z = self.Ntheta.y
            x = self.monitor.plane_distance
        elif self.monitor.plane_axis == 1:
            x = self.Ntheta.x
            z = self.Ntheta.y
            y = self.monitor.plane_distance
        elif self.monitor.plane_axis == 2:
            x = self.Ntheta.x
            y = self.Ntheta.y
            z = self.monitor.plane_distance

        x, y, z = [np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)]

        eta = np.atleast_1d(self.eta(frequencies, medium))
        wave_number = np.atleast_1d(self.k(frequencies, medium))

        e_theta = -(self.Lphi.values + eta[None, None, None, ...] * self.Ntheta.values)
        e_phi = self.Ltheta.values - eta[None, None, None, ...] * self.Nphi.values

        Ex_data = np.zeros((len(x), len(y), len(z), len(frequencies)), dtype=complex)
        Ey_data = np.zeros_like(Ex_data)
        Ez_data = np.zeros_like(Ex_data)

        Hx_data = np.zeros_like(Ex_data)
        Hy_data = np.zeros_like(Ex_data)
        Hz_data = np.zeros_like(Ex_data)

        for i, _x in enumerate(x):
            for j, _y in enumerate(y):
                for k, _z in enumerate(z):
                    r, theta, phi = self.car_2_sph(_x, _y, _z)
                    phase = -1j * wave_number * np.exp(1j * wave_number * r) / (4 * np.pi * r)

                    Et = -phase * e_theta[i, j, k, :]
                    Ep = phase * e_phi[i, j, k, :]
                    Er = np.zeros_like(Et)

                    Ht = -Ep / eta
                    Hp = Et / eta
                    Hr = np.zeros_like(Hp)

                    e_fields = self.sph_2_car_field(Er, Et, Ep, theta, phi)
                    h_fields = self.sph_2_car_field(Hr, Ht, Hp, theta, phi)

                    Ex_data[i, j, k, :] = e_fields[0]
                    Ey_data[i, j, k, :] = e_fields[1]
                    Ez_data[i, j, k, :] = e_fields[2]

                    Hx_data[i, j, k, :] = h_fields[0]
                    Hy_data[i, j, k, :] = h_fields[1]
                    Hz_data[i, j, k, :] = h_fields[2]

        dims = ("x", "y", "z", "f")
        coords = {"x": x, "y": y, "z": z, "f": frequencies}

        Ex = xr.DataArray(data=Ex_data, coords=coords, dims=dims)
        Ey = xr.DataArray(data=Ey_data, coords=coords, dims=dims)
        Ez = xr.DataArray(data=Ez_data, coords=coords, dims=dims)

        Hx = xr.DataArray(data=Hx_data, coords=coords, dims=dims)
        Hy = xr.DataArray(data=Hy_data, coords=coords, dims=dims)
        Hz = xr.DataArray(data=Hz_data, coords=coords, dims=dims)

        field_data = xr.Dataset({"Ex": Ex, "Ey": Ey, "Ez": Ez, "Hx": Hx, "Hy": Hy, "Hz": Hz})

        return field_data

    # def power_cartesian(self, x: ArrayLikeN2F, y: ArrayLikeN2F, z: ArrayLikeN2F) -> xr.DataArray:
    #     """Get power scattered to a point relative to the local origin in cartesian coordinates.

    #     Parameters
    #     ----------
    #     x : Union[float, Tuple[float, ...], np.ndarray]
    #         (micron) x distances relative to the local origin.
    #     y : Union[float, Tuple[float, ...], np.ndarray]
    #         (micron) y distances relative to the local origin.
    #     z : Union[float, Tuple[float, ...], np.ndarray]
    #         (micron) z distances relative to the local origin.

    #     Returns
    #     -------
    #     power : xarray.DataArray
    #         Power at points relative to the local origin.
    #     """

    #     x, y, z = [np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z)]

    #     power_data = np.zeros((len(x), len(y), len(z)))

    #     for i in track(np.arange(len(x)), description="Computing far field power"):
    #         _x = x[i]
    #         for j in np.arange(len(y)):
    #             _y = y[j]
    #             for k in np.arange(len(z)):
    #                 _z = z[k]

    #                 r, theta, phi = self._car_2_sph(_x, _y, _z)
    #                 power_data[i, j, k] = self.power_spherical(r, theta, phi).values

    #     dims = ("x", "y", "z")
    #     coords = {"x": x, "y": y, "z": z}

    #     return xr.DataArray(data=power_data, coords=coords, dims=dims)


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
    def fields_spherical(self, medium: Medium = Medium(permittivity=1)) -> xr.Dataset:
        """Get fields in spherical coordinates relative to the monitor's local origin
        for all k-space points and frequencies specified in :class:`Near2FarKSpaceMonitor`.

        Parameters
        ----------
        medium : :class:`.Medium`
            Background medium in which to radiate near fields to far fields.
            Default: free space.

        Returns
        -------
        xarray.Dataset
            xarray dataset containing (Er, Etheta, Ephi), (Hr, Htheta, Hphi)
            in polar coordinates.
        """

        # Assumes that frequencies and angles are the same for all radiation vectors
        ux = self.Ntheta.ux
        uy = self.Ntheta.uy
        frequencies = self.Ntheta.f

        eta = np.array([self.eta(frequency, medium) for frequency in frequencies])

        # assemble E felds
        eta = eta[None, None, ...]
        Et_array = -(self.Lphi.values + eta * self.Ntheta.values)
        Ep_array = self.Ltheta.values - eta * self.Nphi.values
        Er_array = np.zeros_like(Ep_array)

        dims = ("ux", "uy", "f")
        coords = {"ux": ux, "uy": uy, "f": frequencies}

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


Near2FarDataTypes = (Near2FarAngleData, Near2FarCartesianData, Near2FarKSpaceData)
Near2FarDataType = Union[Near2FarDataTypes]
DATA_TYPE_MAP = DATA_TYPE_MAP.update(
    {data.__fields__["monitor"].type_: data for data in Near2FarDataTypes}
)

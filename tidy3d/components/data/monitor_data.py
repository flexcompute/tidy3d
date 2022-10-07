""" Monitor Level Data, store the DataArrays associated with a single monitor."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Tuple
import warnings
import xarray as xr
import numpy as np
import pydantic as pd

from .data_array import DiffractionDataArray
from ..base import TYPE_TAG_STR
from ..types import Coordinate, Symmetry
from ..grid.grid import Grid
from ..validators import enforce_monitor_fields_present, required_if_symmetry_present
from ..monitor import MonitorType, FieldMonitor, FieldTimeMonitor, ModeSolverMonitor
from ..monitor import ModeMonitor, FluxMonitor, FluxTimeMonitor, PermittivityMonitor
from ..monitor import Near2FarAngleMonitor, Near2FarCartesianMonitor, Near2FarKSpaceMonitor
from ..monitor import DiffractionMonitor
from ..medium import Medium
from ...log import SetupError, log
from ...constants import ETA_0, C_0, MICROMETER
from .dataset import Dataset, AbstractFieldDataset
from .dataset import FieldDataset, FieldTimeDataset, ModeSolverDataset, PermittivityDataset
from .dataset import ModeDataset, FluxDataset, FluxTimeDataset
from .dataset import AbstractNear2FarDataset, Near2FarKSpaceDataset
from .dataset import Near2FarAngleDataset, Near2FarCartesianDataset
from .dataset import DiffractionDataset


class MonitorData(Dataset, ABC):
    """Abstract base class of objects that store data pertaining to a single :class:`.monitor`."""

    monitor: MonitorType = pd.Field(
        ...,
        title="Monitor",
        description="Monitor associated with the data.",
        descriminator=TYPE_TAG_STR,
    )

    @property
    def symmetry_expanded_copy(self) -> MonitorData:
        """Return copy of self with symmetry applied."""
        return self.copy()


class AbstractFieldData(MonitorData, AbstractFieldDataset, ABC):
    """Collection of scalar fields with some symmetry properties."""

    monitor: Union[FieldMonitor, FieldTimeMonitor, PermittivityMonitor, ModeSolverMonitor]

    symmetry: Tuple[Symmetry, Symmetry, Symmetry] = pd.Field(
        (0, 0, 0),
        title="Symmetry",
        description="Symmetry eigenvalues of the original simulation in x, y, and z.",
    )

    symmetry_center: Coordinate = pd.Field(
        None,
        title="Symmetry Center",
        description="Center of the symmetry planes of the original simulation in x, y, and z. "
        "Required only if any of the ``symmetry`` field are non-zero.",
    )
    grid_expanded: Grid = pd.Field(
        None,
        title="Expanded Grid",
        description=":class:`.Grid` on which the symmetry will be expanded. "
        "Required only if any of the ``symmetry`` field are non-zero.",
    )

    _require_sym_center = required_if_symmetry_present("symmetry_center")
    _require_grid_expanded = required_if_symmetry_present("grid_expanded")

    @property
    def symmetry_expanded_copy(self) -> AbstractFieldData:
        """Create a copy of the :class:`.AbstractFieldData` with fields expanded based on symmetry.

        Returns
        -------
        :class:`AbstractFieldData`
            A data object with the symmetry expanded fields.
        """

        if all(sym == 0 for sym in self.symmetry):
            return self.copy()

        new_fields = {}

        for field_name, scalar_data in self.field_components.items():

            grid_key = self.grid_locations[field_name]
            eigenval_fn = self.symmetry_eigenvalues[field_name]

            # get grid locations for this field component on the expanded grid
            grid_locations = self.grid_expanded[grid_key]

            for sym_dim, (sym_val, sym_loc) in enumerate(zip(self.symmetry, self.symmetry_center)):

                dim_name = "xyz"[sym_dim]

                # Continue if no symmetry along this dimension
                if sym_val == 0:
                    continue

                # Get coordinates for this field component on the expanded grid
                coords = grid_locations.to_list[sym_dim]

                # Get indexes of coords that lie on the left of the symmetry center
                flip_inds = np.where(coords < sym_loc)[0]

                # Get the symmetric coordinates on the right
                coords_interp = np.copy(coords)
                coords_interp[flip_inds] = 2 * sym_loc - coords[flip_inds]

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


class FieldData(FieldDataset, AbstractFieldData):
    """Data associated with a :class:`.FieldMonitor`: scalar components of E and H fields.

    Example
    -------
    >>> from tidy3d import ScalarFieldDataArray
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

    _contains_monitor_fields = enforce_monitor_fields_present()


class FieldTimeData(FieldTimeDataset, AbstractFieldData):
    """Data associated with a :class:`.FieldTimeMonitor`: scalar components of E and H fields.

    Example
    -------
    >>> from tidy3d import ScalarFieldTimeDataArray
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

    _contains_monitor_fields = enforce_monitor_fields_present()


class ModeSolverData(ModeSolverDataset, AbstractFieldData):
    """Data associated with a :class:`.ModeSolverMonitor`: scalar components of E and H fields.

    Example
    -------
    >>> from tidy3d import ModeSpec
    >>> from tidy3d import ScalarModeFieldDataArray, ModeIndexDataArray
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


class PermittivityData(PermittivityDataset, AbstractFieldData):
    """Data for a :class:`.PermittivityMonitor`: diagonal components of the permittivity tensor.

    Example
    -------
    >>> from tidy3d import ScalarFieldDataArray
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


class ModeData(MonitorData, ModeDataset):
    """Data associated with a :class:`.ModeMonitor`: modal amplitudes and propagation indices.

    Example
    -------
    >>> from tidy3d import ModeSpec
    >>> from tidy3d import ModeAmpsDataArray, ModeIndexDataArray
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


class FluxData(MonitorData, FluxDataset):
    """Data associated with a :class:`.FluxMonitor`: flux data in the frequency-domain.

    Example
    -------
    >>> from tidy3d import FluxDataArray
    >>> f = [2e14, 3e14]
    >>> coords = dict(f=f)
    >>> flux_data = FluxDataArray(np.random.random(2), coords=coords)
    >>> monitor = FluxMonitor(size=(2,0,6), freqs=[2e14, 3e14], name='flux')
    >>> data = FluxData(monitor=monitor, flux=flux_data)
    """

    monitor: FluxMonitor


class FluxTimeData(MonitorData, FluxTimeDataset):
    """Data associated with a :class:`.FluxTimeMonitor`: flux data in the time-domain.

    Example
    -------
    >>> from tidy3d import FluxTimeDataArray
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(t=t)
    >>> flux_data = FluxTimeDataArray(np.random.random(3), coords=coords)
    >>> monitor = FluxTimeMonitor(size=(2,0,6), interval=100, name='flux_time')
    >>> data = FluxTimeData(monitor=monitor, flux=flux_data)
    """

    monitor: FluxTimeMonitor


class AbstractNear2FarData(MonitorData, AbstractNear2FarDataset, ABC):
    """Collection of radiation vectors in the frequency domain."""

    monitor: Union[Near2FarAngleMonitor, Near2FarCartesianMonitor, Near2FarKSpaceMonitor] = None

    medium: Medium = pd.Field(
        Medium(),
        title="Background Medium",
        description="Background medium in which to radiate near fields to far fields.",
    )

    @staticmethod
    def propagation_factor(medium: Medium, frequency: float) -> complex:
        """Complex valued wavenumber associated with a frequency."""
        index_n, index_k = medium.nk_model(frequency=frequency)
        return (2 * np.pi * frequency / C_0) * (index_n + 1j * index_k)

    @property
    def nk(self) -> Tuple[float, float]:
        """Returns the real and imaginary parts of the background medium's refractive index."""
        return self.medium.nk_model(frequency=self.f)

    @property
    def k(self) -> complex:
        """Returns the complex wave number associated with the background medium."""
        return self.propagation_factor(medium=self.medium, frequency=self.f)

    @property
    def eta(self) -> complex:
        """Returns the complex wave impedance associated with the background medium."""
        eps_complex = self.medium.eps_model(frequency=self.f)
        return ETA_0 / np.sqrt(eps_complex)

    @property
    def rad_vecs_to_fields(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute fields from radiation vectors."""
        eta = self.eta
        e_theta = -(self.Lphi.values + eta * self.Ntheta.values)
        e_phi = self.Ltheta.values - eta * self.Nphi.values
        return e_theta, e_phi

    def propagation_phase(self, dist: Union[float, None]) -> complex:
        """Phase associated with propagation of a distance with a given wavenumber."""
        if dist is None:
            return 1.0
        return -1j * self.k * np.exp(1j * self.k * dist) / (4 * np.pi * dist)

    def fields_sph(self, r: float = None) -> xr.Dataset:
        """Get fields in spherical coordinates relative to the monitor's local origin
        for all angles and frequencies specified in :class:`Near2FarAngleMonitor`.
        If the radial distance ``r`` is provided, a corresponding phase factor is applied
        to the returned fields.

        Parameters
        ----------
        r : float = None
            (micron) radial distance relative to the monitor's local origin.

        Returns
        -------
        ``xarray.Dataset``
            xarray dataset containing
            (``E_r``, ``E_theta``, ``E_phi``, ``H_r``, ``H_theta``, ``H_phi``)
            in polar coordinates.
        """

        # assemble E felds
        e_theta, e_phi = self.rad_vecs_to_fields
        phase = self.propagation_phase(dist=r)
        Et_array = phase * e_theta
        Ep_array = phase * e_phi
        Er_array = np.zeros_like(Ep_array)

        # assemble H fields
        eta = self.eta[None, None, :]
        Ht_array = -Ep_array / eta
        Hp_array = Et_array / eta
        Hr_array = np.zeros_like(Hp_array)

        keys = ("E_r", "E_theta", "E_phi", "H_r", "H_theta", "H_phi")
        vals = (Er_array, Et_array, Ep_array, Hr_array, Ht_array, Hp_array)
        return self.make_dataset(keys=keys, vals=vals)

    @abstractmethod
    def fields(self, r: float = None) -> xr.Dataset:
        """Get fields in spherical coordinates relative to the monitor's local origin
        for all angles and frequencies specified in :class:`Near2FarAngleMonitor`.
        If the radial distance ``r`` is provided, a corresponding phase factor is applied
        to the returned fields.

        Parameters
        ----------
        r : float = None
            (micron) radial distance relative to the monitor's local origin.

        Returns
        -------
            xarray dataset containing
            (``E_r``, ``E_theta``, ``E_phi``, ``H_r``, ``H_theta``, ``H_phi``)
            in polar coordinates.
        """

    @abstractmethod
    def power(self, r: float = None) -> xr.DataArray:
        """Get power measured on the observation grid defined in spherical coordinates.

        Parameters
        ----------
        r : float = None
            (micron) radial distance relative to the local origin.

        Returns
        -------
        ``xarray.DataArray``
            Power at points relative to the local origin.
        """


class Near2FarAngleData(AbstractNear2FarData, Near2FarAngleDataset):
    """Data associated with a :class:`.Near2FarAngleMonitor`: components of radiation vectors.

    Example
    -------
    >>> from tidy3d import Near2FarAngleDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(theta=theta, phi=phi, f=f)
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

    _contains_monitor_fields = enforce_monitor_fields_present()

    def fields(self, r: float = None) -> xr.Dataset:
        """Get fields in spherical coordinates relative to the monitor's local origin
        for all angles and frequencies specified in :class:`Near2FarAngleMonitor`.
        If the radial distance ``r`` is provided, a corresponding phase factor is applied
        to the returned fields.

        Parameters
        ----------
        r : float = None
            (micron) radial distance relative to the monitor's local origin.

        Returns
        -------
        ``xarray.Dataset``
            xarray dataset containing
            (``E_r``, ``E_theta``, ``E_phi``, ``H_r``, ``H_theta``, ``H_phi``)
            in polar coordinates.
        """
        return self.fields_sph(r=r)

    def radar_cross_section(self) -> xr.DataArray:
        """Radar cross section at the observation grid in units of incident power."""

        _, index_k = self.nk
        if not np.all(index_k == 0):
            raise SetupError("Can't compute RCS for a lossy background medium.")

        k = self.k[None, None, ...]
        eta = self.eta[None, None, ...]

        constant = k**2 / (8 * np.pi * eta)
        e_theta, e_phi = self.rad_vecs_to_fields
        rcs_data = constant * (np.abs(e_theta) ** 2 + np.abs(e_phi) ** 2)

        return self.make_data_array(data=rcs_data)

    def power(self, r: float = None) -> xr.DataArray:
        """Get power measured on the observation grid defined in spherical coordinates.

        Parameters
        ----------
        r : float
            (micron) radial distance relative to the local origin.

        Returns
        -------
        ``xarray.DataArray``
            Power at points relative to the local origin.
        """

        if r is None:
            raise ValueError("'r' required by 'Near2FarAngleData.power'")

        field_data = self.fields(r=r)
        Et, Ep = [field_data[comp].values for comp in ["E_theta", "E_phi"]]
        Ht, Hp = [field_data[comp].values for comp in ["H_theta", "H_phi"]]
        power_theta = 0.5 * np.real(Et * np.conj(Hp))
        power_phi = 0.5 * np.real(-Ep * np.conj(Ht))
        power_data = power_theta + power_phi

        return self.make_data_array(data=power_data)


class Near2FarCartesianData(AbstractNear2FarData, Near2FarCartesianDataset):
    """Data associated with a :class:`.Near2FarCartesianMonitor`: components of radiation vectors.

    Example
    -------
    >>> from tidy3d import Near2FarCartesianDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> x = np.linspace(0, 5, 10)
    >>> y = np.linspace(0, 10, 20)
    >>> coords = dict(x=x, y=y, f=f)
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

    _contains_monitor_fields = enforce_monitor_fields_present()

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
    def fields(self, r: float = None) -> xr.Dataset:
        """Get fields on a cartesian plane at a distance relative to monitor center
        along a given axis in cartesian coordinates.

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
        fields_sph = self.fields_sph(r=r_values)
        Er, Et, Ep = (fields_sph[key].values for key in ("E_r", "E_theta", "E_phi"))
        Hr, Ht, Hp = (fields_sph[key].values for key in ("H_r", "H_theta", "H_phi"))

        # convert the field components to cartesian coordinate system
        e_data = self.monitor.sph_2_car_field(Er, Et, Ep, thetas, phis)
        h_data = self.monitor.sph_2_car_field(Hr, Ht, Hp, thetas, phis)

        # package into dataset
        keys = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        field_components = np.concatenate((e_data, h_data), axis=0)
        return self.make_dataset(keys=keys, vals=field_components)

    def power(self, r: float = None) -> xr.Dataset:
        """Get power on the observation grid defined in Cartesian coordinates.

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
        fields_sph = self.fields_sph(r=r_values)
        Et, Ep = (fields_sph[key].values for key in ("E_theta", "E_phi"))
        Ht, Hp = (fields_sph[key].values for key in ("H_theta", "H_phi"))

        # compute power
        power_theta = 0.5 * np.real(Et * np.conj(Hp))
        power_phi = 0.5 * np.real(-Ep * np.conj(Ht))
        power = power_theta + power_phi

        # package as data array
        return self.make_data_array(data=power)


class Near2FarKSpaceData(AbstractNear2FarData, Near2FarKSpaceDataset):
    """Data associated with a :class:`.Near2FarKSpaceMonitor`: components of radiation vectors.

    Example
    -------
    >>> from tidy3d import Near2FarKSpaceDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> ux = np.linspace(0, 5, 10)
    >>> uy = np.linspace(0, 10, 20)
    >>> coords = dict(ux=ux, uy=uy, f=f)
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

    _contains_monitor_fields = enforce_monitor_fields_present()

    # pylint:disable=too-many-locals
    def fields(self, r: float = None) -> xr.Dataset:
        """Get fields in spherical coordinates relative to the monitor's local origin
        for all k-space points and frequencies specified in :class:`Near2FarKSpaceMonitor`.

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
        eta = self.eta[None, None, ...]
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

    def power(self, r: float = None) -> xr.Dataset:
        """Get power on the observation grid defined in k-space.

        Returns
        -------
        ``xarray.Dataset``
            xarray dataset containing power.
        """

        if r is not None:
            log.warning("'r' supplied to 'Near2FarKSpaceData.fields' will not be used.")

        fields = self.fields(r=None)
        Et, Ep, Ht, Hp = (fields[key].values for key in ("E_theta", "E_phi", "H_theta", "H_phi"))

        power_theta = 0.5 * np.real(Et * np.conj(Hp))
        power_phi = 0.5 * np.real(-Ep * np.conj(Ht))
        power_values = power_theta + power_phi

        return self.make_data_array(data=power_values)


class DiffractionData(MonitorData, DiffractionDataset):
    """Data for a :class:`.DiffractionMonitor`: complex components of diffracted far fields.

    Example
    -------
    >>> from tidy3d import DiffractionDataArray
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> orders_x = list(range(-4, 5))
    >>> orders_y = list(range(-6, 7))
    >>> pol = ["s", "p"]
    >>> coords = dict(orders_x=orders_x, orders_y=orders_y, polarization=pol, f=f)
    >>> values = (1+1j) * np.random.random((len(orders_x), len(orders_y), len(pol), len(f)))
    >>> field = DiffractionDataArray(values, coords=coords)
    >>> monitor = DiffractionMonitor(
    ...     center=(1,2,3), size=(np.inf,np.inf,0), freqs=f, name='diffraction',
    ...     orders_x=orders_x, orders_y=orders_y
    ... )
    >>> data = DiffractionData(
    ...     monitor=monitor, L=field, N=field, sim_size=[1,1], bloch_vecs=[1,2]
    ... )
    """

    monitor: DiffractionMonitor

    sim_size: Tuple[float, float] = pd.Field(
        ...,
        title="Simulation size",
        description="Simulation sizes in the local x and y directions.",
        units=MICROMETER,
    )

    bloch_vecs: Tuple[float, float] = pd.Field(
        ...,
        title="Bloch vectors",
        description="Bloch vectors along the local x and y directions in units of "
        "``2 * pi / (simulation size along the respective dimension)``.",
    )

    L: DiffractionDataArray = pd.Field(
        ...,
        title="L",
        description="Complex components of the far field radiation vectors associated with the "
        "electric field for each polarization tangential to ``monitor.normal_axis``, "
        "in a local Cartesian coordinate system whose z-axis is ``monitor.normal_axis``.",
    )

    N: DiffractionDataArray = pd.Field(
        ...,
        title="N",
        description="Complex components of the far field radiation vectors associated with the "
        "magnetic field for each polarization tangential to ``monitor.normal_axis``, "
        "in a local Cartesian coordinate system whose z-axis is ``monitor.normal_axis``.",
    )

    @staticmethod
    def shifted_orders(orders: Tuple[int], bloch_vec: float) -> np.ndarray:
        """Diffraction orders shifted by the Bloch vector."""
        return bloch_vec + np.atleast_1d(orders)

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to thier associated data."""
        return dict(L=self.L, N=self.N)

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> DiffractionData:
        """Copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        for field_name, field_data in self.field_components.items():
            src_amps = source_spectrum_fn(field_data.f)
            fields_norm[field_name] = (field_data / src_amps).astype(field_data.dtype)

        return self.copy(update=fields_norm)

    @property
    def medium(self) -> Medium:
        """Medium in which the near fields are recorded and propagated."""
        return self.monitor.medium

    @property
    def frequencies(self) -> np.ndarray:
        """Frequencies associated with ``monitor``."""
        return np.atleast_1d(self.L.f.values)

    @property
    def orders_x(self) -> np.ndarray:
        """Allowed orders along x."""
        return np.atleast_1d(self.L.orders_x.values)

    @property
    def orders_y(self) -> np.ndarray:
        """Allowed orders along y."""
        return np.atleast_1d(self.L.orders_y.values)

    @property
    def wavenumber(self) -> np.ndarray:
        """Wave number at each frequency."""
        epsilon = self.medium.eps_model(self.frequencies)
        return np.real(2.0 * np.pi * self.frequencies / C_0 * np.sqrt(epsilon))

    @property
    def wavelength(self) -> np.ndarray:
        """Wavelength at each frequency."""
        return 2.0 * np.pi / self.wavenumber

    @property
    def eta(self) -> np.ndarray:
        """Wavelength at each frequency."""
        epsilon = self.medium.eps_model(self.frequencies)
        return np.real(ETA_0 / np.sqrt(epsilon))

    @property
    def ux(self) -> np.ndarray:
        """Normalized wave vector along x relative to ``local_origin`` and oriented
        with respect to ``monitor.normal_dir``, normalized by the wave number in the
        background medium."""
        if self.sim_size[0] == 0:
            return np.atleast_2d(0)
        bloch_x = self.shifted_orders(self.orders_x, self.bloch_vecs[0])
        return bloch_x[:, None] * 2.0 * np.pi / self.sim_size[0] / self.wavenumber[None, :]

    @property
    def uy(self) -> np.ndarray:
        """Normalized wave vector along y relative to ``local_origin`` and oriented
        with respect to ``monitor.normal_dir``, normalized by the wave number in the
        background medium."""
        if self.sim_size[1] == 0:
            return np.atleast_2d(0)
        bloch_y = self.shifted_orders(self.orders_y, self.bloch_vecs[1])
        return bloch_y[:, None] * 2.0 * np.pi / self.sim_size[1] / self.wavenumber[None, :]

    def _make_coords_for_pol(self, pol: Tuple[str, str]) -> Dict[str, Union[np.ndarray, List]]:
        """Make a coordinates dictionary for a given pair of polarization names."""
        coords = {}
        coords["orders_x"] = np.atleast_1d(self.orders_x)
        coords["orders_y"] = np.atleast_1d(self.orders_y)
        coords["polarization"] = pol
        coords["f"] = np.array(self.frequencies)
        return coords

    @property
    def angles(self) -> Tuple[xr.DataArray]:
        """The (theta, phi) angles corresponding to each allowed pair of diffraction
        orders storeds as data arrays. Disallowed angles are set to ``np.nan``.
        """
        # some wave number pairs are outside the light cone, leading to warnings from numpy.arcsin
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in arcsin", category=RuntimeWarning
            )
            ux, uy = self.reciprocal_vectors
            thetas, phis = DiffractionMonitor.kspace_2_sph(ux[:, None, :], uy[None, :, :], axis=2)

        coords = self._make_coords_for_pol(["", ""])
        del coords["polarization"]
        theta_data = xr.DataArray(thetas, coords=coords)
        phi_data = xr.DataArray(phis, coords=coords)
        return theta_data, phi_data

    @property
    def amps(self) -> DiffractionDataArray:
        """Complex power amplitude in each order for 's' and 'p' polarizations, normalized so that
        the power carried by the wave of that order and polarization equals ``abs(amps)^2``.
        """
        cos_theta = np.cos(np.nan_to_num(self.angles[0]))
        norm = 1.0 / np.sqrt(2.0 * ETA_0) / np.sqrt(cos_theta)
        amp_theta = self.E_sph.sel(polarization="theta").values * norm
        amp_phi = self.E_sph.sel(polarization="phi").values * norm

        # stack the amplitudes in s- and p-components along a new polarization axis
        return DiffractionDataArray(
            np.stack([amp_phi, amp_theta], axis=2), coords=self._make_coords_for_pol(["s", "p"])
        )

    @property
    def power(self) -> xr.DataArray:
        """Total power in each order, summed over both polarizations."""
        return (np.abs(self.amps) ** 2).sum(dim="polarization")

    def sph_2_car(self, field: DiffractionDataArray) -> DiffractionDataArray:
        """Transform field stored as a :class:`DiffractionDataArray` to Cartesian coordinates,
        assuming they represent plane waves. Angles are restricted to within the light cone;
        other values are set to `nan`."""
        f_theta = field.sel(polarization="theta").values
        f_phi = field.sel(polarization="phi").values
        theta, phi = self.angles
        f_x, f_y, f_z = self.monitor.sph_2_car_field(0, f_theta, f_phi, theta.values, phi.values)
        f_x, f_y, f_z = [np.nan_to_num(fld) for fld in [f_x, f_y, f_z]]

        return DiffractionDataArray(
            np.stack([f_x, f_y, f_z], axis=2), coords=self._make_coords_for_pol(["x", "y", "z"])
        )

    def car_2_sph(self, field: DiffractionDataArray) -> DiffractionDataArray:
        """Transform field stored as a :class:`DiffractionDataArray` to spherical coordinates,
        assuming they represent plane waves. Angles are restricted to within the light cone;
        other values are set to `nan`."""
        f_x = field.sel(polarization="x").values
        f_y = field.sel(polarization="y").values
        theta, phi = self.angles
        f_phi = np.nan_to_num(-np.sin(phi) * f_x + np.cos(phi) * f_y)
        f_theta = np.nan_to_num((f_x * np.cos(phi) + f_y * np.sin(phi)) * np.cos(theta))

        return DiffractionDataArray(
            np.stack([f_theta, f_phi], axis=2), coords=self._make_coords_for_pol(["theta", "phi"])
        )

    # pylint: disable=invalid-name
    @property
    def L_sph(self) -> DiffractionDataArray:
        """Radiation vectors associated with the electric field in spherical coodinates."""
        return self.car_2_sph(self.L)

    # pylint: disable=invalid-name
    @property
    def N_sph(self) -> DiffractionDataArray:
        """Radiation vectors associated with the magnetic field in spherical coordinates."""
        return self.car_2_sph(self.N)

    # pylint: disable=invalid-name
    @property
    def E_sph(self) -> DiffractionDataArray:
        """Far field electric field in spherical coordinates, normalized so that
        ``(0.5 / eta) * abs(E)**2 / cos(theta)`` is the power density for each polarization
        and each order."""
        l_theta = self.L_sph.sel(polarization="theta").values
        l_phi = self.L_sph.sel(polarization="phi").values
        n_theta = self.N_sph.sel(polarization="theta").values
        n_phi = self.N_sph.sel(polarization="phi").values

        e_theta = -(l_phi + self.eta * n_theta) / 4.0 / np.pi
        e_phi = (l_theta - self.eta * n_phi) / 4.0 / np.pi

        return DiffractionDataArray(
            np.stack([e_theta, e_phi], axis=2), coords=self._make_coords_for_pol(["theta", "phi"])
        )

    # pylint: disable=invalid-name
    @property
    def H_sph(self) -> DiffractionDataArray:
        """Far field magnetic field in spherical coordinates."""
        e_theta = self.E_sph.sel(polarization="theta").values
        e_phi = self.E_sph.sel(polarization="phi").values

        h_theta = -e_phi / self.eta
        h_phi = e_theta / self.eta

        return DiffractionDataArray(
            np.stack([h_theta, h_phi], axis=2), coords=self._make_coords_for_pol(["theta", "phi"])
        )

    @property
    def E_car(self) -> DiffractionDataArray:
        """Far field electric field in Cartesian coordinates."""
        return self.sph_2_car(self.E_sph)

    @property
    def H_car(self) -> DiffractionDataArray:
        """Far field magnetic field in Cartesian coordinates."""
        return self.sph_2_car(self.H_sph)

    @property
    def amps(self) -> DiffractionDataArray:
        """Complex power amplitude in each order for 's' and 'p' polarizations, normalized so that
        the power carried by the wave of that order and polarization equals ``abs(amps)^2``.
        """
        cos_theta = np.cos(np.nan_to_num(self.angles[0]))
        norm = 1.0 / np.sqrt(2.0 * self.eta) / np.sqrt(cos_theta)
        amp_theta = self.E_sph.sel(polarization="theta").values * norm
        amp_phi = self.E_sph.sel(polarization="phi").values * norm

        # stack the amplitudes in s- and p-components along a new polarization axis
        return DiffractionDataArray(
            np.stack([amp_phi, amp_theta], axis=2), coords=self._make_coords_for_pol(["s", "p"])
        )

    @property
    def power(self) -> xr.DataArray:
        """Total power in each order, summed over both polarizations."""
        return (np.abs(self.amps) ** 2).sum(dim="polarization")


MonitorDataTypes = (
    FieldData,
    FieldTimeData,
    PermittivityData,
    ModeSolverData,
    ModeData,
    FluxData,
    FluxTimeData,
    Near2FarKSpaceData,
    Near2FarCartesianData,
    Near2FarAngleData,
    DiffractionData,
)

MonitorDataType = Union[MonitorDataTypes]

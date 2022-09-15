""" Monitor Level Data, store the DataArrays associated with a single monitor."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Dict, Callable, List
import xarray as xr
import numpy as np
import pydantic as pd

from .data_array import ScalarFieldDataArray, ScalarFieldTimeDataArray, ScalarModeFieldDataArray
from .data_array import FluxTimeDataArray, FluxDataArray, ModeIndexDataArray, ModeAmpsDataArray
from .data_array import Near2FarCartesianDataArray, Near2FarKSpaceDataArray, Near2FarAngleDataArray
from .data_array import DataArray
from ..base import Tidy3dBaseModel
from ..types import Axis
from ...log import DataError


class Dataset(Tidy3dBaseModel, ABC):
    """Stores a collection of ``DataArray`` objects as fields."""


class AbstractFieldData(Dataset, ABC):
    """Collection of scalar fields with some symmetry properties."""

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
    def symmetry_eigenvalues(self) -> Dict[str, Union[Callable[[Axis], float], None]]:
        """Maps field components to their (positive) symmetry eigenvalues."""

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

        # convert supplied coordinates to array and assign string mapping to them
        supplied_coord_map = {k: np.array(v) for k, v in zip("xyz", (x, y, z)) if v is not None}

        # dict of data arrays to combine in dataset and return
        centered_fields = {}

        # loop through field components
        for field_name, field_data in self.field_components.items():

            # loop through x, y, z dimensions and raise an error if only one element along dim
            for coord_name, coords_supplied in supplied_coord_map.items():
                coord_data = field_data.data.coords[coord_name]
                if coord_data.size == 1:
                    raise DataError(
                        f"colocate given '{coord_name}={coords_supplied}', but "
                        f"data only has one coordinate at '{coord_name}={coord_data.values[0]}'. "
                        "Therefore, can't colocate along this dimension. "
                        f"supply '{coord_name}=None' to skip it."
                    )

            data_array = field_data.data.interp(**supplied_coord_map, kwargs={"bounds_error": True})
            centered_fields[field_name] = data_array

        # combine all centered fields in a dataset
        return xr.Dataset(centered_fields)


ScalarEMField = Union[ScalarFieldDataArray, ScalarFieldTimeDataArray, ScalarModeFieldDataArray]


class ElectromagneticFieldData(AbstractFieldData, ABC):
    """Stores a collection of E and H fields with x, y, z components."""

    Ex: ScalarEMField = pd.Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field.",
    )
    Ey: ScalarEMField = pd.Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field.",
    )
    Ez: ScalarEMField = pd.Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field.",
    )
    Hx: ScalarEMField = pd.Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field.",
    )
    Hy: ScalarEMField = pd.Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field.",
    )
    Hz: ScalarEMField = pd.Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field.",
    )

    @property
    def field_components(self) -> Dict[str, ScalarEMField]:
        """Maps the field components to thier associated data."""
        all_fields = dict(Ex=self.Ex, Ey=self.Ey, Ez=self.Ez, Hx=self.Hx, Hy=self.Hy, Hz=self.Hz)
        return {key: value for key, value in all_fields.items() if value is not None}

    @property
    def grid_locations(self) -> Dict[str, str]:
        """Maps field components to the string key of their grid locations on the yee lattice."""
        return {field_name: field_name for field_name in ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}

    @property
    def symmetry_eigenvalues(self) -> Dict[str, Union[Callable[[Axis], float], None]]:
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
    >>> data_array = xr.DataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> scalar_field = ScalarFieldDataArray(data=data_array)
    >>> data = FieldData(Ex=scalar_field, Hz=scalar_field)
    """

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


class FieldTimeData(ElectromagneticFieldData):
    """Data associated with a :class:`.FieldTimeMonitor`: scalar components of E and H fields.

    Example
    -------
    >>> x = [-1,1]
    >>> y = [-2,0,2]
    >>> z = [-3,-1,1,3]
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(x=x, y=y, z=z, t=t)
    >>> data_array = xr.DataArray(np.random.random((2,3,4,3)), coords=coords)
    >>> scalar_field = ScalarFieldTimeDataArray(data=data_array)
    >>> data = FieldTimeData(Ex=scalar_field, Hz=scalar_field)
    """

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
    >>> data_array = xr.DataArray((1+1j)*np.random.random((2,1,4,2,5)), coords=field_coords)
    >>> fld = ScalarModeFieldDataArray(data=data_array)
    >>> index_coords = dict(f=f, mode_index=mode_index)
    >>> index_data_array = xr.DataArray((1+1j) * np.random.random((2,5)), coords=index_coords)
    >>> index_data = ModeIndexDataArray(data=index_data_array)
    >>> data = ModeSolverData(Ex=fld, Ey=fld, Ez=fld, Hx=fld, Hy=fld, Hz=fld, n_complex=index_data)
    """

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
    def n_eff(self):
        """Real part of the propagation index."""
        return self.n_complex.data.real

    @property
    def k_eff(self):
        """Imaginary part of the propagation index."""
        return self.n_complex.data.imag

    def sel_mode_index(self, mode_index: pd.NonNegativeInt) -> FieldData:
        """Return :class:`.FieldData` with the specificed mode index selected from the data."""

        scalar_fields = {}
        for field_name, scalar_field_data in self.field_components.items():
            data_array = scalar_field_data.data.sel(mode_index=mode_index).drop_vars("mode_index")
            scalar_fields[field_name] = ScalarFieldDataArray(data=data_array)

        return FieldData(**scalar_fields)

    def plot_field(self, *args, **kwargs):
        """Warn user to use the ``plot_field`` function from :class:`.ModeSolver` instead."""
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
    >>> data_array = xr.DataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> scalar_field = ScalarFieldDataArray(data=data_array)
    >>> data = PermittivityData(eps_xx=scalar_field, eps_yy=scalar_field, eps_zz=scalar_field)
    """

    @property
    def field_components(self) -> Dict[str, ScalarFieldDataArray]:
        """Maps the field components to thier associated data."""
        return dict(eps_xx=self.eps_xx, eps_yy=self.eps_yy, eps_zz=self.eps_zz)

    @property
    def grid_locations(self) -> Dict[str, str]:
        """Maps field components to the string key of their grid locations on the yee lattice."""
        return dict(eps_xx="Ex", eps_yy="Ey", eps_zz="Ez")

    @property
    def symmetry_eigenvalues(self) -> Dict[str, Union[Callable[[Axis], float], None]]:
        """Maps field components to their (positive) symmetry eigenvalues."""
        return dict(eps_xx=None, eps_yy=None, eps_zz=None)

    eps_xx: ScalarFieldDataArray = pd.Field(
        ...,
        title="Epsilon xx",
        description="Spatial distribution of the xx-component of the relative permittivity tensor.",
    )
    eps_yy: ScalarFieldDataArray = pd.Field(
        ...,
        title="Epsilon yy",
        description="Spatial distribution of the yt-component of the relative permittivity tensor.",
    )
    eps_zz: ScalarFieldDataArray = pd.Field(
        ...,
        title="Epsilon zz",
        description="Spatial distribution of the zz-component of the relative permittivity tensor.",
    )


class ModeData(Dataset):
    """Data associated with a :class:`.ModeMonitor`: modal amplitudes and propagation indices.

    Example
    -------
    >>> from tidy3d import ModeSpec
    >>> direction = ["+", "-"]
    >>> f = [1e14, 2e14, 3e14]
    >>> mode_index = np.arange(5)
    >>> index_coords = dict(f=f, mode_index=mode_index)
    >>> index_data_array = xr.DataArray((1+1j) * np.random.random((3, 5)), coords=index_coords)
    >>> index_data = ModeIndexDataArray(data=index_data_array)
    >>> amp_coords = dict(direction=direction, f=f, mode_index=mode_index)
    >>> amp_data_array = xr.DataArray((1+1j) * np.random.random((2, 3, 5)), coords=amp_coords)
    >>> amp_data = ModeAmpsDataArray(data=amp_data_array)
    >>> data = ModeData(amps=amp_data, n_complex=index_data)
    """

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
        return self.n_complex.data.real

    @property
    def k_eff(self):
        """Imaginary part of the propagation index."""
        return self.n_complex.data.imag


class FluxData(Dataset):
    """Data associated with a :class:`.FluxMonitor`: flux data in the frequency-domain.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> coords = dict(f=f)
    >>> flux_data_array = xr.DataArray(np.random.random(2), coords=coords)
    >>> flux_data = FluxDataArray(data=flux_data_array)
    >>> data = FluxData(flux=flux_data)
    """

    flux: FluxDataArray


class FluxTimeData(Dataset):
    """Data associated with a :class:`.FluxTimeMonitor`: flux data in the time-domain.

    Example
    -------
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(t=t)
    >>> flux_data_array = xr.DataArray(np.random.random(3), coords=coords)
    >>> flux_data = FluxTimeDataArray(data=flux_data_array)
    >>> data = FluxTimeData(flux=flux_data)
    """

    flux: FluxTimeDataArray


N2FDataArray = Union[Near2FarAngleDataArray, Near2FarCartesianDataArray, Near2FarKSpaceDataArray]


class AbstractNear2FarData(Dataset, ABC):
    """Collection of radiation vectors in the frequency domain."""

    Ntheta: N2FDataArray = pd.Field(
        None,
        title="Ntheta",
        description="Spatial distribution of the theta-component of the 'N' radiation vector.",
    )
    Nphi: N2FDataArray = pd.Field(
        None,
        title="Nphi",
        description="Spatial distribution of phi-component of the 'N' radiation vector.",
    )
    Ltheta: N2FDataArray = pd.Field(
        None,
        title="Ltheta",
        description="Spatial distribution of theta-component of the 'L' radiation vector.",
    )
    Lphi: N2FDataArray = pd.Field(
        None,
        title="Lphi",
        description="Spatial distribution of phi-component of the 'L' radiation vector.",
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
    def sph_2_car(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
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
    def sph_2_car_field(
        f_r: float, f_theta: float, f_phi: float, theta: float, phi: float
    ) -> Tuple[complex, complex, complex]:
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
    def kspace_2_sph(ux: float, uy: float, axis: Axis) -> Tuple[float, float]:
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
        if axis == 2:
            return theta_local, phi_local

        x = np.cos(theta_local)
        y = np.sin(theta_local) * np.sin(phi_local)
        z = -np.sin(theta_local) * np.cos(phi_local)

        if axis == 1:
            x, y, z = -z, x, -y

        theta = np.arccos(z)
        phi = np.arctan2(y, x)
        return theta, phi

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
    >>> coords = dict(theta=theta, phi=phi, f=f)
    >>> values = (1+1j) * np.random.random((len(theta), len(phi), len(f)))
    >>> data_array = xr.DataArray(values, coords=coords)
    >>> scalar_field = Near2FarAngleDataArray(data=data_array)
    >>> data = Near2FarAngleData(
    ...     Ntheta=scalar_field,
    ...     Nphi=scalar_field,
    ...     Ltheta=scalar_field,
    ...     Lphi=scalar_field
    ... )
    """

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
    >>> coords = dict(x=x, y=y, f=f)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(f)))
    >>> data_array = xr.DataArray(values, coords=coords)
    >>> scalar_field = Near2FarCartesianDataArray(data=data_array)
    >>> data = Near2FarCartesianData(
    ...     Ntheta=scalar_field,
    ...     Nphi=scalar_field,
    ...     Ltheta=scalar_field,
    ...     Lphi=scalar_field,
    ... )
    """

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

    @property
    def x(self) -> np.ndarray:
        """X positions."""
        return self.Ntheta.x.values

    @property
    def y(self) -> np.ndarray:
        """Y positions."""
        return self.Ntheta.y.values

    def spherical_coords(
        self, plane_dist: float, plane_axis: Axis
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The data coordinates in spherical coordinate system."""
        xs, ys, _ = np.meshgrid(self.x, self.y, np.array([0]), indexing="ij")
        zs = plane_distance * np.ones_like(xs)
        coords = [xs, ys]
        coords.insert(plane_axis, zs)
        x_glob, y_glob, z_glob = coords
        return self.car_2_sph(x_glob, y_glob, z_glob)

    # pylint:disable=too-many-arguments, too-many-locals
    def fields(
        self,
        plane_dist: float,
        plane_axis: Axis,
        r: float = None,
        medium: Medium = Medium(permittivity=1),
    ) -> xr.Dataset:
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
        r_values, thetas, phis = self.spherical_coords(plane_dist=plane_dist, plane_axis=plane_axis)
        fields_sph = self.fields_sph(r=r_values, medium=medium)
        Er, Et, Ep = (fields_sph[key].values for key in ("E_r", "E_theta", "E_phi"))
        Hr, Ht, Hp = (fields_sph[key].values for key in ("H_r", "H_theta", "H_phi"))

        # convert the field components to cartesian coordinate system
        e_data = self.sph_2_car_field(Er, Et, Ep, thetas, phis)
        h_data = self.sph_2_car_field(Hr, Ht, Hp, thetas, phis)

        # package into dataset
        keys = ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")
        field_components = np.concatenate((e_data, h_data), axis=0)
        return self.make_dataset(keys=keys, vals=field_components)

    def power(
        self,
        plane_dist: float,
        plane_axis: Axis,
        r: float = None,
        medium: Medium = Medium(permittivity=1),
    ) -> xr.Dataset:
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
        r_values, _, _ = self.spherical_coords(plane_dist=plane_dist, plane_axis=plane_axis)
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
    >>> coords = dict(ux=ux, uy=uy, f=f)
    >>> values = (1+1j) * np.random.random((len(ux), len(uy), len(f)))
    >>> data_array = xr.DataArray(values, coords=coords)
    >>> scalar_field = Near2FarKSpaceDataArray(data=data_array)
    >>> data = Near2FarKSpaceData(
    ...     Ntheta=scalar_field,
    ...     Nphi=scalar_field,
    ...     Ltheta=scalar_field,
    ...     Lphi=scalar_field
    ... )
    """

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


DatasetTypes = (
    FieldData,
    FieldTimeData,
    PermittivityData,
    ModeSolverData,
    ModeData,
    FluxData,
    FluxTimeData,
    Near2FarAngleData,
    Near2FarCartesianData,
    Near2FarKSpaceData,
)

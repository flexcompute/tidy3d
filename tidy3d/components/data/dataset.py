"""Collections of DataArrays."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Dict, Tuple, Callable, List
import xarray as xr
import numpy as np
import pydantic as pd

from .data_array import ScalarFieldDataArray, ScalarFieldTimeDataArray, ScalarModeFieldDataArray
from .data_array import FluxTimeDataArray, FluxDataArray, ModeIndexDataArray, ModeAmpsDataArray
from .data_array import Near2FarAngleDataArray, Near2FarCartesianDataArray, Near2FarKSpaceDataArray
from .data_array import DataArray, DiffractionDataArray

from ..base import Tidy3dBaseModel
from ..types import Axis
from ...log import DataError
from ...constants import C_0, MICROMETER


class Dataset(Tidy3dBaseModel, ABC):
    """Abstract base class for objects that store collections of `:class:`.DataArray`s."""

    # pylint:disable=unused-argument
    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> Dataset:
        """Return copy of self after normalization is applied using source spectrum function."""
        return self.copy()


class AbstractFieldDataset(Dataset, ABC):
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
    def symmetry_eigenvalues(self) -> Dict[str, Callable[[Axis], float]]:
        """Maps field components to their (positive) symmetry eigenvalues."""

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


EMSCALARFIELDTYPE = Union[ScalarFieldDataArray, ScalarFieldTimeDataArray, ScalarModeFieldDataArray]


class ElectromagneticFieldDataset(AbstractFieldDataset, ABC):
    """Stores a collection of E and H fields with x, y, z components."""

    Ex: EMSCALARFIELDTYPE = pd.Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field.",
    )
    Ey: EMSCALARFIELDTYPE = pd.Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field.",
    )
    Ez: EMSCALARFIELDTYPE = pd.Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field.",
    )
    Hx: EMSCALARFIELDTYPE = pd.Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field.",
    )
    Hy: EMSCALARFIELDTYPE = pd.Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field.",
    )
    Hz: EMSCALARFIELDTYPE = pd.Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field.",
    )

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to thier associated data."""
        fields = dict(Ex=self.Ex, Ey=self.Ey, Ez=self.Ez, Hx=self.Hx, Hy=self.Hy, Hz=self.Hz)
        return {field_name: field for field_name, field in fields.items() if field is not None}

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


class FieldDataset(ElectromagneticFieldDataset):
    """Dataset storing a collection of the scalar components of E and H fields in the freq. domain

    Example
    -------
    >>> x = [-1,1]
    >>> y = [-2,0,2]
    >>> z = [-3,-1,1,3]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> scalar_field = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> data = FieldDataset(Ex=scalar_field, Hz=scalar_field)
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

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> FieldDataset:
        """Return copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        for field_name, field_data in self.field_components.items():
            src_amps = source_spectrum_fn(field_data.f)
            fields_norm[field_name] = field_data / src_amps

        return self.copy(update=fields_norm)


class FieldTimeDataset(ElectromagneticFieldDataset):
    """Dataset storing a collection of the scalar components of E and H fields in the time domain

    Example
    -------
    >>> x = [-1,1]
    >>> y = [-2,0,2]
    >>> z = [-3,-1,1,3]
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(x=x, y=y, z=z, t=t)
    >>> scalar_field = ScalarFieldTimeDataArray(np.random.random((2,3,4,3)), coords=coords)
    >>> data = FieldTimeDataset(Ex=scalar_field, Hz=scalar_field)
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


class ModeSolverDataset(ElectromagneticFieldDataset):
    """Dataset storing scalar components of E and H fields as a function of freq. and mode_index.

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
    >>> data = ModeSolverDataset(
    ...     Ex=field,
    ...     Ey=field,
    ...     Ez=field,
    ...     Hx=field,
    ...     Hy=field,
    ...     Hz=field,
    ...     n_complex=index_data
    ... )
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

    def sel_mode_index(self, mode_index: pd.NonNegativeInt) -> FieldDataset:
        """Return :class:`.FieldData` for the specificed mode index."""

        fields = {}
        for field_name, data in self.field_components.items():
            data = data.sel(mode_index=mode_index)
            coords = {dim: data.coords[dim] for dim in "xyzf"}
            scalar_field = ScalarFieldDataArray(data.data, coords=coords)
            fields[field_name] = scalar_field

        return FieldDataset(**fields)

    def plot_field(self, *args, **kwargs):
        """Warn user to use the :class:`.ModeSolver` ``plot_field`` function now."""
        raise DeprecationWarning(
            "The 'plot_field()' method was moved to the 'ModeSolver' object."
            "Once the 'ModeSolver' is contructed, one may call '.plot_field()' on the object and "
            "the modes will be computed and displayed with 'Simulation' overlay."
        )


class PermittivityDataset(AbstractFieldDataset):
    """Dataset storing the diagonal components of the permittivity tensor.

    Example
    -------
    >>> x = [-1,1]
    >>> y = [-2,0,2]
    >>> z = [-3,-1,1,3]
    >>> f = [2e14, 3e14]
    >>> coords = dict(x=x, y=y, z=z, f=f)
    >>> sclr_fld = ScalarFieldDataArray((1+1j) * np.random.random((2,3,4,2)), coords=coords)
    >>> data = PermittivityDataset(eps_xx=sclr_fld, eps_yy=sclr_fld, eps_zz=sclr_fld)
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


class ModeDataset(Dataset):
    """Dataset storing modal amplitudes and propagation indices.

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
    >>> data = ModeDataset(amps=amp_data, n_complex=index_data)
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
        return self.n_complex.real

    @property
    def k_eff(self):
        """Imaginary part of the propagation index."""
        return self.n_complex.imag

    def normalize(self, source_spectrum_fn) -> ModeDataset:
        """Return copy of self after normalization is applied using source spectrum function."""
        if self.amps is None:
            raise DataError("ModeData contains no amp data, can't normalize.")
        source_freq_amps = source_spectrum_fn(self.amps.f)[None, :, None]
        return self.copy(update={"amps": self.amps / source_freq_amps})


class FluxDataset(Dataset):
    """Dataset storing flux through a surface in the frequency domain.

    Example
    -------
    >>> f = [2e14, 3e14]
    >>> coords = dict(f=f)
    >>> flux_data = FluxDataArray(np.random.random(2), coords=coords)
    >>> data = FluxDataset(flux=flux_data)
    """

    flux: FluxDataArray

    def normalize(self, source_spectrum_fn) -> FluxDataset:
        """Return copy of self after normalization is applied using source spectrum function."""
        source_freq_amps = source_spectrum_fn(self.flux.f)
        source_power = abs(source_freq_amps) ** 2
        return self.copy(update={"flux": self.flux / source_power})


class FluxTimeDataset(Dataset):
    """Dataset storing flux through a surface in the time domain.

    Example
    -------
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(t=t)
    >>> flux_data = FluxTimeDataArray(np.random.random(3), coords=coords)
    >>> data = FluxTimeDataset(flux=flux_data)
    """

    flux: FluxTimeDataArray


RADVECTYPE = Union[Near2FarAngleDataArray, Near2FarCartesianDataArray, Near2FarKSpaceDataArray]


class AbstractNear2FarDataset(Dataset, ABC):
    """Collection of radiation vectors in the frequency domain."""

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

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> AbstractNear2FarDataset:
        """Return copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        for field_name, field_data in self.field_components.items():
            src_amps = source_spectrum_fn(field_data.f)
            fields_norm[field_name] = field_data / src_amps

        return self.copy(update=fields_norm)


class Near2FarAngleDataset(AbstractNear2FarDataset):
    """Stores radiation vectors as function of angles in spherical coordinates.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> theta = np.linspace(0, np.pi, 10)
    >>> phi = np.linspace(0, 2*np.pi, 20)
    >>> coords = dict(theta=theta, phi=phi, f=f)
    >>> values = (1+1j) * np.random.random((len(theta), len(phi), len(f)))
    >>> scalar_field = Near2FarAngleDataArray(values, coords=coords)
    >>> data = Near2FarAngleDataset(
    ...     Ntheta=scalar_field, Nphi=scalar_field,
    ...     Ltheta=scalar_field, Lphi=scalar_field
    ... )
    """

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

    @property
    def theta(self) -> np.ndarray:
        """Polar angles."""
        return self.Ntheta.theta.values

    @property
    def phi(self) -> np.ndarray:
        """Azimuthal angles."""
        return self.Ntheta.phi.values


class Near2FarCartesianDataset(AbstractNear2FarDataset):
    """Stores radiation vectors in cartesian coordinates.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> x = np.linspace(0, 5, 10)
    >>> y = np.linspace(0, 10, 20)
    >>> coords = dict(x=x, y=y, f=f)
    >>> values = (1+1j) * np.random.random((len(x), len(y), len(f)))
    >>> scalar_field = Near2FarCartesianDataArray(values, coords=coords)
    >>> data = Near2FarCartesianDataset(
    ...     Ntheta=scalar_field, Nphi=scalar_field,
    ...     Ltheta=scalar_field, Lphi=scalar_field
    ... )
    """

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

    @property
    def x(self) -> np.ndarray:
        """X positions."""
        return self.Ntheta.x.values

    @property
    def y(self) -> np.ndarray:
        """Y positions."""
        return self.Ntheta.y.values


class Near2FarKSpaceDataset(AbstractNear2FarDataset):
    """Stores radiation vectors in reciprocal space coordinates.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> ux = np.linspace(0, 5, 10)
    >>> uy = np.linspace(0, 10, 20)
    >>> coords = dict(ux=ux, uy=uy, f=f)
    >>> values = (1+1j) * np.random.random((len(ux), len(uy), len(f)))
    >>> scalar_field = Near2FarKSpaceDataArray(values, coords=coords)
    >>> data = Near2FarKSpaceDataset(
    ...     Ntheta=scalar_field, Nphi=scalar_field,
    ...     Ltheta=scalar_field, Lphi=scalar_field
    ... )
    """

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

    @property
    def ux(self) -> np.ndarray:
        """reciprocal X positions."""
        return self.Ntheta.ux.values

    @property
    def uy(self) -> np.ndarray:
        """reciprocal Y positions."""
        return self.Ntheta.uy.values


# pylint: disable=too-many-public-methods
class DiffractionDataset(Dataset):
    """Stores amplitudes associated with diffraction orders.

    Example
    -------
    >>> f = np.linspace(1e14, 2e14, 10)
    >>> orders_x = list(range(-4, 5))
    >>> orders_y = list(range(-6, 7))
    >>> pol = ["s", "p"]
    >>> coords = dict(orders_x=orders_x, orders_y=orders_y, polarization=pol, f=f)
    >>> values = (1+1j) * np.random.random((len(orders_x), len(orders_y), len(pol), len(f)))
    >>> field = DiffractionDataArray(values, coords=coords)
    >>> data = DiffractionDataset(L=field, N=field, sim_size=[1,1], bloch_vecs=[1,2])
    """

    sim_size: Tuple[float, float] = pd.Field(
        ...,
        title="Domain size",
        description="Size of the near field in the local x and y directions.",
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
        "electric field for each polarization tangential to the normal axis, "
        "in a local Cartesian coordinate system whose z-axis is normal to the near field plane.",
    )

    N: DiffractionDataArray = pd.Field(
        ...,
        title="N",
        description="Complex components of the far field radiation vectors associated with the "
        "electric field for each polarization tangential to the normal axis, "
        "in a local Cartesian coordinate system whose z-axis is normal to the near field plane.",
    )

    @staticmethod
    def shifted_orders(orders: Tuple[int], bloch_vec: float) -> np.ndarray:
        """Diffraction orders shifted by the Bloch vector."""
        return bloch_vec + np.atleast_1d(orders)

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to thier associated data."""
        return dict(L=self.L, N=self.N)

    def normalize(self, source_spectrum_fn: Callable[[float], complex]) -> DiffractionDataset:
        """Copy of self after normalization is applied using source spectrum function."""
        fields_norm = {}
        for field_name, field_data in self.field_components.items():
            src_amps = source_spectrum_fn(field_data.f)
            fields_norm[field_name] = field_data / src_amps

        return self.copy(update=fields_norm)

    @property
    def frequencies(self) -> np.ndarray:
        """``np.ndarray`` of frequencies in the dataset."""
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
    def wavelength(self) -> np.ndarray:
        """Wavelength at each frequency."""
        return C_0 / self.frequencies

    @property
    def wavenumber(self) -> np.ndarray:
        """Wave number at each frequency."""
        return 2.0 * np.pi / self.wavelength

    def reciprocal_coords(self, orders: np.ndarray, size: float, bloch_vec: float) -> np.ndarray:
        """Get the normalized "u" reciprocal coords for a vector of orders, size, and bloch vec."""
        if size == 0:
            return np.atleast_2d(0)
        bloch_array = self.shifted_orders(orders, bloch_vec)
        return bloch_array[:, None] * 2.0 * np.pi / size / self.wavenumber[None, :]

    @property
    def reciprocal_vectors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the normalized "ux" and "uy" reciprocal vectors."""
        return (self.ux, self.uy)

    @property
    def ux(self) -> np.ndarray:
        """Normalized wave vector along x relative to ``local_origin`` and oriented
        with respect to ``monitor.normal_dir``, normalized by the wave number in the
        background medium."""
        return self.reciprocal_coords(
            orders=self.orders_x, size=self.sim_size[0], bloch_vec=self.bloch_vecs[0]
        )

    @property
    def uy(self) -> np.ndarray:
        """Normalized wave vector along y relative to ``local_origin`` and oriented
        with respect to ``monitor.normal_dir``, normalized by the wave number in the
        background medium."""
        return self.reciprocal_coords(
            orders=self.orders_y, size=self.sim_size[1], bloch_vec=self.bloch_vecs[1]
        )

    def _make_coords_for_pol(self, pol: Tuple[str, str]) -> Dict[str, Union[np.ndarray, List]]:
        """Make a coordinates dictionary for a given pair of polarization names."""
        coords = {}
        coords["orders_x"] = np.atleast_1d(self.orders_x)
        coords["orders_y"] = np.atleast_1d(self.orders_y)
        coords["polarization"] = pol
        coords["f"] = np.array(self.frequencies)
        return coords

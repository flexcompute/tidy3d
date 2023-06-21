"""Collections of DataArrays."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Dict, Callable, Any

import xarray as xr
import numpy as np
import pydantic as pd

from .data_array import DataArray
from .data_array import ScalarFieldDataArray, ScalarFieldTimeDataArray, ScalarModeFieldDataArray
from .data_array import ModeIndexDataArray
from .data_array import TriangleMeshDataArray

from ..base import Tidy3dBaseModel
from ..types import Axis
from ...exceptions import DataError


class Dataset(Tidy3dBaseModel, ABC):
    """Abstract base class for objects that store collections of `:class:`.DataArray`s."""


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

    def package_colocate_results(self, centered_fields: Dict[str, ScalarFieldDataArray]) -> Any:
        """How to package the dictionary of fields computed via self.colocate()."""
        return xr.Dataset(centered_fields)

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
                coord_data = np.array(field_data.coords[coord_name])
                if coord_data.size == 1:
                    raise DataError(
                        f"colocate given {coord_name}={coords_supplied}, but "
                        f"data only has one coordinate at {coord_name}={coord_data[0]}. "
                        "Therefore, can't colocate along this dimension. "
                        f"supply {coord_name}=None to skip it."
                    )

            centered_fields[field_name] = field_data.interp(
                **supplied_coord_map, kwargs={"bounds_error": True}
            )

        # combine all centered fields in a dataset
        return self.package_colocate_results(centered_fields)


EMScalarFieldType = Union[ScalarFieldDataArray, ScalarFieldTimeDataArray, ScalarModeFieldDataArray]


class ElectromagneticFieldDataset(AbstractFieldDataset, ABC):
    """Stores a collection of E and H fields with x, y, z components."""

    Ex: EMScalarFieldType = pd.Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field.",
    )
    Ey: EMScalarFieldType = pd.Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field.",
    )
    Ez: EMScalarFieldType = pd.Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field.",
    )
    Hx: EMScalarFieldType = pd.Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field.",
    )
    Hy: EMScalarFieldType = pd.Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field.",
    )
    Hz: EMScalarFieldType = pd.Field(
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

    n_group: ModeIndexDataArray = pd.Field(
        None,
        title="Group Index",
        description="Index associated with group velocity of the mode.",
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
        description="Spatial distribution of the xx-component of the relative permittivity.",
    )
    eps_yy: ScalarFieldDataArray = pd.Field(
        ...,
        title="Epsilon yy",
        description="Spatial distribution of the yy-component of the relative permittivity.",
    )
    eps_zz: ScalarFieldDataArray = pd.Field(
        ...,
        title="Epsilon zz",
        description="Spatial distribution of the zz-component of the relative permittivity.",
    )


class TriangleMeshDataset(Dataset):
    """Dataset for storing triangular surface data."""

    surface_mesh: TriangleMeshDataArray = pd.Field(
        ...,
        title="Surface mesh data",
        description="Dataset containing the surface triangles and corresponding face indices "
        "for a surface mesh.",
    )

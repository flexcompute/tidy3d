"""Collections of DataArrays."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Union, get_args

import numpy as np
import pydantic.v1 as pd
import xarray as xr

from ...constants import C_0, PICOSECOND_PER_NANOMETER_PER_KILOMETER, UnitScaling
from ...exceptions import DataError
from ...log import log
from ..base import Tidy3dBaseModel
from ..types import Axis, xyz
from .data_array import (
    DataArray,
    EMEScalarFieldDataArray,
    EMEScalarModeFieldDataArray,
    GroupIndexDataArray,
    ModeDispersionDataArray,
    ModeIndexDataArray,
    ScalarFieldDataArray,
    ScalarFieldTimeDataArray,
    ScalarModeFieldCylindricalDataArray,
    ScalarModeFieldDataArray,
    TimeDataArray,
    TriangleMeshDataArray,
)
from .zbf import ZBFData

DEFAULT_MAX_SAMPLES_PER_STEP = 10_000
DEFAULT_MAX_CELLS_PER_STEP = 10_000
DEFAULT_TOLERANCE_CELL_FINDING = 1e-6


class Dataset(Tidy3dBaseModel, ABC):
    """Abstract base class for objects that store collections of `:class:`.DataArray`s."""


class AbstractFieldDataset(Dataset, ABC):
    """Collection of scalar fields with some symmetry properties."""

    @property
    @abstractmethod
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to their associated data."""

    def apply_phase(self, phase: float) -> AbstractFieldDataset:
        """Create a copy where all elements are phase-shifted by a value (in radians)."""
        if phase == 0.0:
            return self
        phasor = np.exp(1j * phase)
        field_components_shifted = {}
        for fld_name, fld_cmp in self.field_components.items():
            fld_cmp_shifted = phasor * fld_cmp
            field_components_shifted[fld_name] = fld_cmp_shifted
        return self.updated_copy(**field_components_shifted)

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
        """Colocate all of the data at a set of x, y, z coordinates.

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
            For more details refer to `xarray's Documentation <https://tinyurl.com/cyca3krz>`_.

        Note
        ----
        For many operations (such as flux calculations and plotting),
        it is important that the fields are colocated at the same spatial locations.
        Be sure to apply this method to your field data in those cases.
        """

        if hasattr(self, "monitor") and self.monitor.colocate:
            with log as consolidated_logger:
                consolidated_logger.warning(
                    "Colocating data that has already been colocated during the solver "
                    "run. For most accurate results when colocating to custom coordinates set "
                    "'Monitor.colocate' to 'False' to use the raw data on the Yee grid "
                    "and avoid double interpolation. Note: the default value was changed to 'True' "
                    "in Tidy3D version 2.4.0."
                )

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


EMScalarFieldType = Union[
    ScalarFieldDataArray,
    ScalarFieldTimeDataArray,
    ScalarModeFieldDataArray,
    ScalarModeFieldCylindricalDataArray,
    EMEScalarModeFieldDataArray,
    EMEScalarFieldDataArray,
]


class ElectromagneticFieldDataset(AbstractFieldDataset, ABC):
    """Stores a collection of E and H fields with x, y, z components."""

    Ex: Optional[EMScalarFieldType] = pd.Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field.",
    )
    Ey: Optional[EMScalarFieldType] = pd.Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field.",
    )
    Ez: Optional[EMScalarFieldType] = pd.Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field.",
    )
    Hx: Optional[EMScalarFieldType] = pd.Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field.",
    )
    Hy: Optional[EMScalarFieldType] = pd.Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field.",
    )
    Hz: Optional[EMScalarFieldType] = pd.Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field.",
    )

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to their associated data."""
        fields = {
            "Ex": self.Ex,
            "Ey": self.Ey,
            "Ez": self.Ez,
            "Hx": self.Hx,
            "Hy": self.Hy,
            "Hz": self.Hz,
        }
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

    Ex: Optional[ScalarFieldDataArray] = pd.Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field.",
    )
    Ey: Optional[ScalarFieldDataArray] = pd.Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field.",
    )
    Ez: Optional[ScalarFieldDataArray] = pd.Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field.",
    )
    Hx: Optional[ScalarFieldDataArray] = pd.Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field.",
    )
    Hy: Optional[ScalarFieldDataArray] = pd.Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field.",
    )
    Hz: Optional[ScalarFieldDataArray] = pd.Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field.",
    )

    def from_zbf(filename: str, dim1: xyz, dim2: xyz) -> FieldDataset:
        """Creates a :class:`.FieldDataset` from a Zemax Beam File (``.zbf``).

        Parameters
        ----------
        filename: str
            The file name of the .zbf file to read.
        dim1: xyz
            Tangential field component to map the x-dimension of the zbf data to.
            eg. ``dim1 = "z"`` sets ``FieldDataset.Ez`` to ``Ex`` of the zbf data.
        dim2: xyz
            Tangential field component to map the y-dimension of the zbf data to.
            eg. ``dim2 = "z"`` sets ``FieldDataset.Ez`` to ``Ey`` of the zbf data.

        Returns
        -------
        :class:`.FieldDataset`
            A :class:`.FieldDataset` object with two tangential E field components populated
            by zbf data.

        See Also
        --------
        :class:`.ZBFData`:
            A class containing data read in from a ``.zbf`` file.
        """
        log.warning(
            "'FieldDataset.from_zbf()' is currently an experimental feature."
            " If any issues are encountered, please contact Flexcompute support 'https://www.flexcompute.com/tidy3d/technical-support/'"
        )

        if dim1 not in get_args(xyz):
            raise ValueError(f"'dim1' = '{dim1}' is not allowed, must be one of 'x', 'y', or 'z'.")
        if dim2 not in get_args(xyz):
            raise ValueError(f"'dim2' = '{dim2}' is not allowed, must be one of 'x', 'y', or 'z'.")
        if dim1 == dim2:
            raise ValueError("'dim1' and 'dim2' must be different.")

        # get the third dimension
        dim3 = list(set(get_args(xyz)) - {dim1, dim2})[0]
        dims = {"x": 0, "y": 1, "z": 2}
        dim2expand = dims[dim3]  # this is for expanding E field arrays

        # load zbf data
        zbfdata = ZBFData.read_zbf(filename)

        # Grab E fields, dimensions, wavelength
        edim1 = zbfdata.Ex
        edim2 = zbfdata.Ey
        n1 = zbfdata.nx
        n2 = zbfdata.ny
        d1 = zbfdata.dx / UnitScaling[zbfdata.unit]
        d2 = zbfdata.dy / UnitScaling[zbfdata.unit]
        wavelength = zbfdata.wavelength / UnitScaling[zbfdata.unit]

        # make scalar field data arrays
        len1 = d1 * (n1 - 1)
        len2 = d2 * (n2 - 1)
        coords1 = np.linspace(-len1 / 2, len1 / 2, n1)
        coords2 = np.linspace(-len2 / 2, len2 / 2, n2)
        f = [C_0 / wavelength]
        Edim1 = ScalarFieldDataArray(
            np.expand_dims(edim1, axis=(dim2expand, 3)),
            coords={
                dim1: coords1,
                dim2: coords2,
                dim3: [0],
                "f": f,
            },
        )
        Edim2 = ScalarFieldDataArray(
            np.expand_dims(edim2, axis=(dim2expand, 3)),
            coords={
                dim1: coords1,
                dim2: coords2,
                dim3: [0],
                "f": f,
            },
        )

        return FieldDataset(
            **{
                f"E{dim1}": Edim1,
                f"E{dim2}": Edim2,
            }
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

    Ex: Optional[ScalarFieldTimeDataArray] = pd.Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field.",
    )
    Ey: Optional[ScalarFieldTimeDataArray] = pd.Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field.",
    )
    Ez: Optional[ScalarFieldTimeDataArray] = pd.Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field.",
    )
    Hx: Optional[ScalarFieldTimeDataArray] = pd.Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field.",
    )
    Hy: Optional[ScalarFieldTimeDataArray] = pd.Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field.",
    )
    Hz: Optional[ScalarFieldTimeDataArray] = pd.Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field.",
    )

    def apply_phase(self, phase: float) -> AbstractFieldDataset:
        """Create a copy where all elements are phase-shifted by a value (in radians)."""

        if phase != 0.0:
            raise ValueError("Can't apply phase to time-domain field data, which is real-valued.")

        return self


class AuxFieldDataset(AbstractFieldDataset, ABC):
    """Stores a collection of aux fields with x, y, z components."""

    Nfx: Optional[EMScalarFieldType] = pd.Field(
        None,
        title="Nfx",
        description="Spatial distribution of the free carrier density for "
        "polarization in the x-direction.",
    )
    Nfy: Optional[EMScalarFieldType] = pd.Field(
        None,
        title="Nfy",
        description="Spatial distribution of the free carrier density for "
        "polarization in the y-direction.",
    )
    Nfz: Optional[EMScalarFieldType] = pd.Field(
        None,
        title="Nfz",
        description="Spatial distribution of the free carrier density for "
        "polarization in the z-direction.",
    )

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to their associated data."""
        fields = {
            "Nfx": self.Nfx,
            "Nfy": self.Nfy,
            "Nfz": self.Nfz,
        }
        return {field_name: field for field_name, field in fields.items() if field is not None}

    @property
    def grid_locations(self) -> Dict[str, str]:
        """Maps field components to the string key of their grid locations on the yee lattice."""
        return dict(Nfx="Ex", Nfy="Ey", Nfz="Ez")

    @property
    def symmetry_eigenvalues(self) -> Dict[str, Callable[[Axis], float]]:
        """Maps field components to their (positive) symmetry eigenvalues."""

        return dict(
            Nfx=lambda dim: +1,
            Nfy=lambda dim: +1,
            Nfz=lambda dim: +1,
        )


class AuxFieldTimeDataset(AuxFieldDataset):
    """Dataset storing a collection of the scalar components of aux fields in the time domain

    Example
    -------
    >>> x = [-1,1]
    >>> y = [-2,0,2]
    >>> z = [-3,-1,1,3]
    >>> t = [0, 1e-12, 2e-12]
    >>> coords = dict(x=x, y=y, z=z, t=t)
    >>> scalar_field = ScalarFieldTimeDataArray(np.random.random((2,3,4,3)), coords=coords)
    >>> data = AuxFieldTimeDataset(Nfx=scalar_field)
    """

    Nfx: Optional[ScalarFieldTimeDataArray] = pd.Field(
        None,
        title="Nfx",
        description="Spatial distribution of the free carrier density for polarization "
        "in the x-direction.",
    )
    Nfy: Optional[ScalarFieldTimeDataArray] = pd.Field(
        None,
        title="Nfy",
        description="Spatial distribution of the free carrier density for polarization "
        "in the y-direction.",
    )
    Nfz: Optional[ScalarFieldTimeDataArray] = pd.Field(
        None,
        title="Nfz",
        description="Spatial distribution of the free carrier density for polarization "
        "in the z-direction.",
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

    Ex: Optional[ScalarModeFieldDataArray] = pd.Field(
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field of the mode.",
    )
    Ey: Optional[ScalarModeFieldDataArray] = pd.Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field of the mode.",
    )
    Ez: Optional[ScalarModeFieldDataArray] = pd.Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field of the mode.",
    )
    Hx: Optional[ScalarModeFieldDataArray] = pd.Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field of the mode.",
    )
    Hy: Optional[ScalarModeFieldDataArray] = pd.Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field of the mode.",
    )
    Hz: Optional[ScalarModeFieldDataArray] = pd.Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field of the mode.",
    )

    n_complex: ModeIndexDataArray = pd.Field(
        ...,
        title="Propagation Index",
        description="Complex-valued effective propagation constants associated with the mode.",
    )

    n_group_raw: Optional[GroupIndexDataArray] = pd.Field(
        None,
        alias="n_group",  # This is for backwards compatibility only when loading old data
        title="Group Index",
        description="Index associated with group velocity of the mode.",
    )

    dispersion_raw: Optional[ModeDispersionDataArray] = pd.Field(
        None,
        title="Dispersion",
        description="Dispersion parameter for the mode.",
        units=PICOSECOND_PER_NANOMETER_PER_KILOMETER,
    )

    @property
    def field_components(self) -> Dict[str, DataArray]:
        """Maps the field components to their associated data."""
        fields = {
            "Ex": self.Ex,
            "Ey": self.Ey,
            "Ez": self.Ez,
            "Hx": self.Hx,
            "Hy": self.Hy,
            "Hz": self.Hz,
        }
        return {field_name: field for field_name, field in fields.items() if field is not None}

    @property
    def n_eff(self) -> ModeIndexDataArray:
        """Real part of the propagation index."""
        return self.n_complex.real

    @property
    def k_eff(self) -> ModeIndexDataArray:
        """Imaginary part of the propagation index."""
        return self.n_complex.imag

    @property
    def n_group(self) -> GroupIndexDataArray:
        """Group index."""
        if self.n_group_raw is None:
            log.warning(
                "The group index was not computed. To calculate group index, pass "
                "'group_index_step = True' in the 'ModeSpec'.",
                log_once=True,
            )
        return self.n_group_raw

    @property
    def dispersion(self) -> ModeDispersionDataArray:
        r"""Dispersion parameter.

        .. math::

           D = -\frac{\lambda}{c_0} \frac{{\rm d}^2 n_{\text{eff}}}{{\rm d}\lambda^2}
        """
        if self.dispersion_raw is None:
            log.warning(
                "The dispersion was not computed. To calculate dispersion, pass "
                "'group_index_step = True' in the 'ModeSpec'.",
                log_once=True,
            )
        return self.dispersion_raw

    def plot_field(self, *args, **kwargs):
        """Warn user to use the :class:`.ModeSolver` ``plot_field`` function now."""
        raise DeprecationWarning(
            "The 'plot_field()' method was moved to the 'ModeSolver' object."
            "Once the 'ModeSolver' is constructed, one may call '.plot_field()' on the object and "
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
        """Maps the field components to their associated data."""
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


class TimeDataset(Dataset):
    """Dataset for storing a function of time."""

    values: TimeDataArray = pd.Field(
        ..., title="Values", description="Values as a function of time."
    )

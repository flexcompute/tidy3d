"""Collections of DataArrays."""

from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Tuple, Union

import numpy as np
import pydantic.v1 as pd
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation

from ...constants import PICOSECOND_PER_NANOMETER_PER_KILOMETER, inf
from ...exceptions import DataError, Tidy3dNotImplementedError, ValidationError
from ...log import log
from ...packaging import requires_vtk, vtk
from ..base import Tidy3dBaseModel, cached_property, skip_if_fields_missing
from ..types import ArrayLike, Ax, Axis, Bound, Coordinate, Literal, annotate_type
from ..viz import add_ax_if_none, equal_aspect, plot_params_grid
from .data_array import (
    DATA_ARRAY_MAP,
    CellDataArray,
    DataArray,
    EMEScalarFieldDataArray,
    EMEScalarModeFieldDataArray,
    GroupIndexDataArray,
    IndexedDataArray,
    ModeDispersionDataArray,
    ModeIndexDataArray,
    PointDataArray,
    ScalarFieldDataArray,
    ScalarFieldTimeDataArray,
    ScalarModeFieldDataArray,
    SpatialDataArray,
    TimeDataArray,
    TriangleMeshDataArray,
)

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
    EMEScalarModeFieldDataArray,
    EMEScalarFieldDataArray,
]


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

    def apply_phase(self, phase: float) -> AbstractFieldDataset:
        """Create a copy where all elements are phase-shifted by a value (in radians)."""

        if phase != 0.0:
            raise ValueError("Can't apply phase to time-domain field data, which is real-valued.")

        return self


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
        None,
        title="Ex",
        description="Spatial distribution of the x-component of the electric field of the mode.",
    )
    Ey: ScalarModeFieldDataArray = pd.Field(
        None,
        title="Ey",
        description="Spatial distribution of the y-component of the electric field of the mode.",
    )
    Ez: ScalarModeFieldDataArray = pd.Field(
        None,
        title="Ez",
        description="Spatial distribution of the z-component of the electric field of the mode.",
    )
    Hx: ScalarModeFieldDataArray = pd.Field(
        None,
        title="Hx",
        description="Spatial distribution of the x-component of the magnetic field of the mode.",
    )
    Hy: ScalarModeFieldDataArray = pd.Field(
        None,
        title="Hy",
        description="Spatial distribution of the y-component of the magnetic field of the mode.",
    )
    Hz: ScalarModeFieldDataArray = pd.Field(
        None,
        title="Hz",
        description="Spatial distribution of the z-component of the magnetic field of the mode.",
    )

    n_complex: ModeIndexDataArray = pd.Field(
        ...,
        title="Propagation Index",
        description="Complex-valued effective propagation constants associated with the mode.",
    )

    n_group_raw: GroupIndexDataArray = pd.Field(
        None,
        alias="n_group",  # This is for backwards compatibility only when loading old data
        title="Group Index",
        description="Index associated with group velocity of the mode.",
    )

    dispersion_raw: ModeDispersionDataArray = pd.Field(
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


class UnstructuredGridDataset(Dataset, np.lib.mixins.NDArrayOperatorsMixin, ABC):
    """Abstract base for datasets that store unstructured grid data."""

    points: PointDataArray = pd.Field(
        ...,
        title="Grid Points",
        description="Coordinates of points composing the unstructured grid.",
    )

    values: IndexedDataArray = pd.Field(
        ...,
        title="Point Values",
        description="Values stored at the grid points.",
    )

    cells: CellDataArray = pd.Field(
        ...,
        title="Grid Cells",
        description="Cells composing the unstructured grid specified as connections between grid "
        "points.",
    )

    @property
    def name(self) -> str:
        """Dataset name."""
        # we redirect name to values.name
        return self.values.name

    @property
    def is_complex(self) -> bool:
        """Data type."""
        return np.iscomplexobj(self.values)

    @property
    def _double_type(self):
        """Corresponding double data type."""
        return np.complex128 if self.is_complex else np.float64

    @pd.validator("points", always=True)
    def points_right_dims(cls, val):
        """Check that point coordinates have the right dimensionality."""
        # currently support only the standard axis ordering, that is 01(2)
        axis_coords_expected = np.arange(cls._point_dims())
        axis_coords_given = val.axis.data
        if np.any(axis_coords_given != axis_coords_expected):
            raise ValidationError(
                f"Points array is expected to have {axis_coords_expected} coord values along 'axis'"
                f" (given: {axis_coords_given})."
            )
        return val

    @property
    def is_uniform(self):
        """Whether each element is of equal value in ``values``."""
        return self.values.is_uniform

    @pd.validator("points", always=True)
    def points_right_indexing(cls, val):
        """Check that points are indexed corrrectly."""
        indices_expected = np.arange(len(val.data))
        indices_given = val.index.data
        if np.any(indices_expected != indices_given):
            raise ValidationError(
                "Coordinate 'index' of array 'points' is expected to have values (0, 1, 2, ...). "
                "This can be easily achieved, for example, by using "
                "PointDataArray(data, dims=['index', 'axis'])."
            )
        return val

    @pd.validator("values", always=True)
    def values_right_indexing(cls, val):
        """Check that data values are indexed correctly."""
        # currently support only simple ordered indexing of points, that is, 0, 1, 2, ...
        indices_expected = np.arange(len(val.data))
        indices_given = val.index.data
        if np.any(indices_expected != indices_given):
            raise ValidationError(
                "Coordinate 'index' of array 'values' is expected to have values (0, 1, 2, ...). "
                "This can be easily achieved, for example, by using "
                "IndexedDataArray(data, dims=['index'])."
            )
        return val

    @pd.validator("values", always=True)
    @skip_if_fields_missing(["points"])
    def number_of_values_matches_points(cls, val, values):
        """Check that the number of data values matches the number of grid points."""
        num_values = len(val)

        points = values.get("points")
        num_points = len(points)

        if num_points != num_values:
            raise ValidationError(
                f"The number of data values ({num_values}) does not match the number of grid "
                f"points ({num_points})."
            )
        return val

    @pd.validator("cells", always=True)
    def match_cells_to_vtk_type(cls, val):
        """Check that cell connections does not have duplicate points."""
        if vtk is None:
            return val

        # using val.astype(np.int32/64) directly causes issues when dataarray are later checked ==
        return CellDataArray(val.data.astype(vtk["id_type"], copy=False), coords=val.coords)

    @pd.validator("cells", always=True)
    def cells_right_type(cls, val):
        """Check that cell are of the right type."""
        # only supporting the standard ordering of cell vertices 012(3)
        vertex_coords_expected = np.arange(cls._cell_num_vertices())
        vertex_coords_given = val.vertex_index.data
        if np.any(vertex_coords_given != vertex_coords_expected):
            raise ValidationError(
                f"Cell connections array is expected to have {vertex_coords_expected} coord values"
                f" along 'vertex_index' (given: {vertex_coords_given})."
            )
        return val

    @pd.validator("cells", always=True)
    @skip_if_fields_missing(["points"])
    def check_cell_vertex_range(cls, val, values):
        """Check that cell connections use only defined points."""
        all_point_indices_used = val.data.ravel()
        # skip validation if zero size data
        if len(all_point_indices_used) > 0:
            min_index_used = np.min(all_point_indices_used)
            max_index_used = np.max(all_point_indices_used)

            points = values.get("points")
            num_points = len(points)

            if max_index_used > num_points - 1 or min_index_used < 0:
                raise ValidationError(
                    "Cell connections array uses undefined point indices in the range "
                    f"[{min_index_used}, {max_index_used}]. The valid range of point indices is "
                    f"[0, {num_points-1}]."
                )
        return val

    @classmethod
    def _find_degenerate_cells(cls, cells: CellDataArray):
        """Find explicitly degenerate cells if any.
        That is, cells that use the same point indices for their different vertices.
        """
        indices = cells.data
        # skip validation if zero size data
        degenerate_cell_inds = set()
        if len(indices) > 0:
            for i in range(cls._cell_num_vertices() - 1):
                for j in range(i + 1, cls._cell_num_vertices()):
                    degenerate_cell_inds = degenerate_cell_inds.union(
                        np.where(indices[:, i] == indices[:, j])[0]
                    )

        return degenerate_cell_inds

    @classmethod
    def _remove_degenerate_cells(cls, cells: CellDataArray):
        """Remove explicitly degenerate cells if any.
        That is, cells that use the same point indices for their different vertices.
        """
        degenerate_cells = cls._find_degenerate_cells(cells=cells)
        if len(degenerate_cells) > 0:
            data = np.delete(cells.values, list(degenerate_cells), axis=0)
            cell_index = np.delete(cells.cell_index.values, list(degenerate_cells))
            return CellDataArray(
                data=data, coords=dict(cell_index=cell_index, vertex_index=cells.vertex_index)
            )
        return cells

    @classmethod
    def _remove_unused_points(
        cls, points: PointDataArray, values: IndexedDataArray, cells: CellDataArray
    ):
        """Remove unused points if any.
        That is, points that are not used in any grid cell.
        """

        used_indices = np.unique(cells.values.ravel())
        num_points = len(points)

        if len(used_indices) != num_points or np.any(np.diff(used_indices) != 1):
            min_index = np.min(used_indices)
            map_len = np.max(used_indices) - min_index + 1
            index_map = np.zeros(map_len)
            index_map[used_indices - min_index] = np.arange(len(used_indices))

            cells = CellDataArray(data=index_map[cells.data - min_index], coords=cells.coords)
            points = PointDataArray(points.data[used_indices, :], dims=["index", "axis"])
            values = IndexedDataArray(values.data[used_indices], dims=["index"])

        return points, values, cells

    def clean(self, remove_degenerate_cells=True, remove_unused_points=True):
        """Remove degenerate cells and/or unused points."""
        if remove_degenerate_cells:
            cells = self._remove_degenerate_cells(cells=self.cells)
        else:
            cells = self.cells

        if remove_unused_points:
            points, values, cells = self._remove_unused_points(self.points, self.values, cells)
        else:
            points = self.points
            values = self.values

        return self.updated_copy(points=points, values=values, cells=cells)

    @pd.validator("cells", always=True)
    def warn_degenerate_cells(cls, val):
        """Check that cell connections does not have duplicate points."""
        degenerate_cells = cls._find_degenerate_cells(val)
        num_degenerate_cells = len(degenerate_cells)
        if num_degenerate_cells > 0:
            log.warning(
                f"Unstructured grid contains {num_degenerate_cells} degenerate cell(s). "
                "Such cells can be removed by using function "
                "'.clean(remove_degenerate_cells: bool = True, remove_unused_points: bool = True)'. "
                "For example, 'dataset = dataset.clean()'."
            )
        return val

    @pd.root_validator(pre=True, allow_reuse=True)
    def _warn_if_none(cls, values):
        """Warn if any of data arrays are not loaded."""

        no_data_fields = []
        for field_name in ["points", "cells", "values"]:
            field = values.get(field_name)
            if isinstance(field, str) and field in DATA_ARRAY_MAP.keys():
                no_data_fields.append(field_name)
        if len(no_data_fields) > 0:
            formatted_names = [f"'{fname}'" for fname in no_data_fields]
            log.warning(
                f"Loading {', '.join(formatted_names)} without data. Constructing an empty dataset."
            )
            values["points"] = PointDataArray(
                np.zeros((0, cls._point_dims())), dims=["index", "axis"]
            )
            values["cells"] = CellDataArray(
                np.zeros((0, cls._cell_num_vertices())), dims=["cell_index", "vertex_index"]
            )
            values["values"] = IndexedDataArray(np.zeros(0), dims=["index"])
        return values

    @pd.root_validator(skip_on_failure=True, allow_reuse=True)
    def _warn_unused_points(cls, values):
        """Warn if some points are unused."""
        point_indices = set(np.arange(len(values["values"].data)))
        used_indices = set(values["cells"].values.ravel())

        if not point_indices.issubset(used_indices):
            log.warning(
                "Unstructured grid dataset contains unused points. "
                "Consider calling 'clean()' to remove them."
            )

        return values

    def rename(self, name: str) -> UnstructuredGridDataset:
        """Return a renamed array."""
        return self.updated_copy(values=self.values.rename(name))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Override of numpy functions."""

        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with the same class or a scalar
            if not isinstance(x, (numbers.Number, type(self))):
                return Tidy3dNotImplementedError

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.values if isinstance(x, UnstructuredGridDataset) else x for x in inputs)
        if out:
            kwargs["out"] = tuple(
                x.values if isinstance(x, UnstructuredGridDataset) else x for x in out
            )
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(self.updated_copy(values=x) for x in result)
        elif method == "at":
            # no return value
            return None
        else:
            # one return value
            return self.updated_copy(values=result)

    @property
    def real(self) -> UnstructuredGridDataset:
        """Real part of dataset."""
        return self.updated_copy(values=self.values.real)

    @property
    def imag(self) -> UnstructuredGridDataset:
        """Imaginary part of dataset."""
        return self.updated_copy(values=self.values.imag)

    @property
    def abs(self) -> UnstructuredGridDataset:
        """Absolute value of dataset."""
        return self.updated_copy(values=self.values.abs)

    @cached_property
    def bounds(self) -> Bound:
        """Grid bounds."""
        return tuple(np.min(self.points.data, axis=0)), tuple(np.max(self.points.data, axis=0))

    @classmethod
    @abstractmethod
    def _point_dims(cls) -> pd.PositiveInt:
        """Dimensionality of stored grid point coordinates."""

    @cached_property
    @abstractmethod
    def _points_3d_array(self):
        """3D coordinates of grid points."""

    @classmethod
    @abstractmethod
    def _cell_num_vertices(cls) -> pd.PositiveInt:
        """Number of vertices in a cell."""

    @classmethod
    @abstractmethod
    @requires_vtk
    def _vtk_cell_type(cls):
        """VTK cell type to use in the VTK representation."""

    @cached_property
    def _vtk_offsets(self) -> ArrayLike:
        """Offsets array to use in the VTK representation."""
        offsets = np.arange(len(self.cells) + 1) * self._cell_num_vertices()
        if vtk is None:
            return offsets

        return offsets.astype(vtk["id_type"], copy=False)

    @property
    @requires_vtk
    def _vtk_cells(self):
        """VTK cell array to use in the VTK representation."""
        cells = vtk["mod"].vtkCellArray()
        cells.SetData(
            vtk["numpy_to_vtkIdTypeArray"](self._vtk_offsets),
            vtk["numpy_to_vtkIdTypeArray"](self.cells.data.ravel()),
        )
        return cells

    @property
    @requires_vtk
    def _vtk_points(self):
        """VTK point array to use in the VTK representation."""
        pts = vtk["mod"].vtkPoints()
        pts.SetData(vtk["numpy_to_vtk"](self._points_3d_array))
        return pts

    @property
    @requires_vtk
    def _vtk_obj(self):
        """A VTK representation (vtkUnstructuredGrid) of the grid."""

        grid = vtk["mod"].vtkUnstructuredGrid()

        grid.SetPoints(self._vtk_points)
        grid.SetCells(self._vtk_cell_type(), self._vtk_cells)
        if self.is_complex:
            # vtk doesn't support complex numbers
            # so we will store our complex array as a two-component vtk array
            data_values = self.values.values.view("(2,)float")
        else:
            data_values = self.values.values
        point_data_vtk = vtk["numpy_to_vtk"](data_values)
        point_data_vtk.SetName(self.values.name)
        grid.GetPointData().AddArray(point_data_vtk)

        return grid

    @requires_vtk
    def _plane_slice_raw(self, axis: Axis, pos: float):
        """Slice data with a plane and return the resulting VTK object."""

        if pos > self.bounds[1][axis] or pos < self.bounds[0][axis]:
            raise DataError(
                f"Slicing plane (axis: {axis}, pos: {pos}) does not intersect the unstructured grid "
                f"(extent along axis {axis}: {self.bounds[0][axis]}, {self.bounds[1][axis]})."
            )

        origin = [0, 0, 0]
        origin[axis] = pos

        normal = [0, 0, 0]
        # orientation of normal is important for edge (literally) cases
        normal[axis] = -1
        if pos > (self.bounds[0][axis] + self.bounds[1][axis]) / 2:
            normal[axis] = 1

        # create cutting plane
        plane = vtk["mod"].vtkPlane()
        plane.SetOrigin(origin[0], origin[1], origin[2])
        plane.SetNormal(normal[0], normal[1], normal[2])

        # create cutter
        cutter = vtk["mod"].vtkPlaneCutter()
        cutter.SetPlane(plane)
        cutter.SetInputData(self._vtk_obj)
        cutter.InterpolateAttributesOn()
        cutter.Update()

        # clean up the slice
        cleaner = vtk["mod"].vtkCleanPolyData()
        cleaner.SetInputData(cutter.GetOutput())
        cleaner.Update()

        return cleaner.GetOutput()

    @abstractmethod
    @requires_vtk
    def plane_slice(
        self, axis: Axis, pos: float
    ) -> Union[SpatialDataArray, UnstructuredGridDataset]:
        """Slice data with a plane and return the Tidy3D representation of the result
        (``UnstructuredGridDataset``).

        Parameters
        ----------
        axis : Axis
            The normal direction of the slicing plane.
        pos : float
            Position of the slicing plane along its normal direction.

        Returns
        -------
        Union[SpatialDataArray, UnstructuredGridDataset]
            The resulting slice.
        """

    @staticmethod
    @requires_vtk
    def _read_vtkUnstructuredGrid(fname: str):
        """Load a :class:`vtkUnstructuredGrid` from a file."""
        reader = vtk["mod"].vtkXMLUnstructuredGridReader()
        reader.SetFileName(fname)
        reader.Update()
        grid = reader.GetOutput()

        return grid

    @classmethod
    @abstractmethod
    @requires_vtk
    def _from_vtk_obj(
        cls,
        vtk_obj,
        field: str = None,
        remove_degenerate_cells: bool = False,
        remove_unused_points: bool = False,
    ) -> UnstructuredGridDataset:
        """Initialize from a vtk object."""

    @classmethod
    @requires_vtk
    def from_vtu(
        cls,
        file: str,
        field: str = None,
        remove_degenerate_cells: bool = False,
        remove_unused_points: bool = False,
    ) -> UnstructuredGridDataset:
        """Load unstructured data from a vtu file.

        Parameters
        ----------
        fname : str
            Full path to the .vtu file to load the unstructured data from.
        field : str = None
            Name of the field to load.
        remove_degenerate_cells : bool = False
            Remove explicitly degenerate cells.
        remove_unused_points : bool = False
            Remove unused points.

        Returns
        -------
        UnstructuredGridDataset
            Unstructured data.
        """
        grid = cls._read_vtkUnstructuredGrid(file)
        return cls._from_vtk_obj(
            grid,
            field=field,
            remove_degenerate_cells=remove_degenerate_cells,
            remove_unused_points=remove_unused_points,
        )

    @requires_vtk
    def to_vtu(self, fname: str):
        """Exports unstructured grid data into a .vtu file.

        Parameters
        ----------
        fname : str
            Full path to the .vtu file to save the unstructured data to.
        """

        writer = vtk["mod"].vtkXMLUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(self._vtk_obj)
        writer.Write()

    @classmethod
    @requires_vtk
    def _get_values_from_vtk(
        cls, vtk_obj, num_points: pd.PositiveInt, field: str = None
    ) -> IndexedDataArray:
        """Get point data values from a VTK object."""

        point_data = vtk_obj.GetPointData()
        num_point_arrays = point_data.GetNumberOfArrays()

        if num_point_arrays == 0:
            log.warning(
                "No point data is found in a VTK object. '.values' will be initialized to zeros."
            )
            values_numpy = np.zeros(num_points)
            values_name = None

        else:
            if field is not None:
                array_vtk = point_data.GetAbstractArray(field)
            else:
                array_vtk = point_data.GetAbstractArray(0)

            # currently we assume there is only one point data array provided in the VTK object
            if num_point_arrays > 1 and field is None:
                array_name = array_vtk.GetName()
                log.warning(
                    f"{num_point_arrays} point data arrays are found in a VTK object. "
                    f"Only the first array (name: {array_name}) will be used to initialize "
                    "'.values' while the rest will be ignored."
                )

            # currently we assume data is real or complex scalar
            num_components = array_vtk.GetNumberOfComponents()
            if num_components > 2:
                raise DataError(
                    "Found point data array in a VTK object is expected to have maximum 2 "
                    "components (1 is for real data, 2 is for complex data). "
                    f"Found {num_components} components."
                )

            # check that number of values matches number of grid points
            num_tuples = array_vtk.GetNumberOfTuples()
            if num_tuples != num_points:
                raise DataError(
                    f"The length of found point data array ({num_tuples}) does not match the number"
                    f" of grid points ({num_points})."
                )

            values_numpy = vtk["vtk_to_numpy"](array_vtk)
            values_name = array_vtk.GetName()

            # vtk doesn't support complex numbers
            # we store our complex array as a two-component vtk array
            # so here we convert that into a single component complex array
            if num_components == 2:
                values_numpy = values_numpy.view("complex")[:, 0]

        values = IndexedDataArray(
            values_numpy, coords=dict(index=np.arange(len(values_numpy))), name=values_name
        )

        return values

    @requires_vtk
    def box_clip(self, bounds: Bound) -> UnstructuredGridDataset:
        """Clip the unstructured grid using a box defined by ``bounds``.

        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        UnstructuredGridDataset
            Clipped grid.
        """

        # make and run a VTK clipper
        clipper = vtk["mod"].vtkBoxClipDataSet()
        clipper.SetOrientation(0)
        clipper.SetBoxClip(
            bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1], bounds[0][2], bounds[1][2]
        )
        clipper.SetInputData(self._vtk_obj)
        clipper.GenerateClipScalarsOn()
        clipper.GenerateClippedOutputOff()
        clipper.Update()
        clip = clipper.GetOutput()

        # clean grid from unused points
        grid_cleaner = vtk["mod"].vtkRemoveUnusedPoints()
        grid_cleaner.SetInputData(clip)
        grid_cleaner.GenerateOriginalPointIdsOff()
        grid_cleaner.Update()
        clean_clip = grid_cleaner.GetOutput()

        # no intersection check
        if clean_clip.GetNumberOfPoints() == 0:
            raise DataError("Clipping box does not intersect the unstructured grid.")

        return self._from_vtk_obj(
            clean_clip, remove_degenerate_cells=True, remove_unused_points=True
        )

    def interp(
        self,
        x: Union[float, ArrayLike],
        y: Union[float, ArrayLike],
        z: Union[float, ArrayLike],
        fill_value: Union[float, Literal["extrapolate"]] = None,
        use_vtk: bool = False,
        method: Literal["linear", "nearest"] = "linear",
        max_samples_per_step: int = DEFAULT_MAX_SAMPLES_PER_STEP,
        max_cells_per_step: int = DEFAULT_MAX_CELLS_PER_STEP,
        rel_tol: float = DEFAULT_TOLERANCE_CELL_FINDING,
    ) -> SpatialDataArray:
        """Interpolate data at provided x, y, and z.

        Parameters
        ----------
        x : Union[float, ArrayLike]
            x-coordinates of sampling points.
        y : Union[float, ArrayLike]
            y-coordinates of sampling points.
        z : Union[float, ArrayLike]
            z-coordinates of sampling points.
        fill_value : Union[float, Literal["extrapolate"]] = 0
            Value to use when filling points without interpolated values. If ``"extrapolate"`` then
            nearest values are used. Note: in a future version the default value will be changed
            to ``"extrapolate"``.
        use_vtk : bool = False
            Use vtk's interpolation functionality or Tidy3D's own implementation. Note: this
            option will be removed in a future version.
        method: Literal["linear", "nearest"] = "linear"
            Interpolation method to use.
        max_samples_per_step : int = 1e4
            Max number of points to interpolate at per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        max_cells_per_step : int = 1e4
            Max number of cells to interpolate from per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        rel_tol : float = 1e-6
            Relative tolerance when determining whether a point belongs to a cell.

        Returns
        -------
        SpatialDataArray
            Interpolated data.
        """

        if fill_value is None:
            log.warning(
                "Default parameter setting 'fill_value=0' will be changed to "
                "'fill_value=``extrapolate``' in a future version."
            )
            fill_value = 0

        # calculate the resulting array shape
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        if method == "nearest":
            interpolated_values = self._interp_nearest(x=x, y=y, z=z)
        else:
            if fill_value == "extrapolate":
                fill_value_actual = np.nan
            else:
                fill_value_actual = fill_value

            if use_vtk:
                log.warning("Note that option 'use_vtk=True' will be removed in future versions.")
                interpolated_values = self._interp_vtk(x=x, y=y, z=z, fill_value=fill_value_actual)
            else:
                interpolated_values = self._interp_py(
                    x=x,
                    y=y,
                    z=z,
                    fill_value=fill_value_actual,
                    max_samples_per_step=max_samples_per_step,
                    max_cells_per_step=max_cells_per_step,
                    rel_tol=rel_tol,
                )

            if fill_value == "extrapolate" and method != "nearest":
                interpolated_values = self._fill_nans_from_nearests(
                    interpolated_values, x=x, y=y, z=z
                )

        return SpatialDataArray(
            interpolated_values, coords=dict(x=x, y=y, z=z), name=self.values.name
        )

    def _interp_nearest(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
    ) -> ArrayLike:
        """Interpolate data at provided x, y, and z using Scipy's nearest neighbor interpolator.

        Parameters
        ----------
        x : ArrayLike
            x-coordinates of sampling points.
        y : ArrayLike
            y-coordinates of sampling points.
        z : ArrayLike
            z-coordinates of sampling points.

        Returns
        -------
        ArrayLike
            Interpolated data.
        """
        from scipy.interpolate import NearestNDInterpolator

        # use scipy's nearest neighbor interpolator
        X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
        interp = NearestNDInterpolator(self._points_3d_array, self.values.values)
        values = interp(X, Y, Z)

        return values

    def _fill_nans_from_nearests(
        self,
        values: ArrayLike,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
    ) -> ArrayLike:
        """Replace nan's in ``values`` with nearest data points.

        Parameters
        ----------
        values : ArrayLike
            3D array containing nan's
        x : ArrayLike
            x-coordinates of sampling points.
        y : ArrayLike
            y-coordinates of sampling points.
        z : ArrayLike
            z-coordinates of sampling points.

        Returns
        -------
        ArrayLike
            Data without nan's.
        """

        # locate all nans
        nans = np.isnan(values)

        if np.sum(nans) > 0:
            from scipy.interpolate import NearestNDInterpolator

            # use scipy's nearest neighbor interpolator
            X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
            interp = NearestNDInterpolator(self._points_3d_array, self.values.values)
            values_to_replace_nans = interp(X[nans], Y[nans], Z[nans])
            values[nans] = values_to_replace_nans

        return values

    @requires_vtk
    def _interp_vtk(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        fill_value: float,
    ) -> ArrayLike:
        """Interpolate data at provided x, y, and z using vtk package.

        Parameters
        ----------
        x : Union[float, ArrayLike]
            x-coordinates of sampling points.
        y : Union[float, ArrayLike]
            y-coordinates of sampling points.
        z : Union[float, ArrayLike]
            z-coordinates of sampling points.
        fill_value : float = 0
            Value to use when filling points without interpolated values.

        Returns
        -------
        ArrayLike
            Interpolated data.
        """

        shape = (len(x), len(y), len(z))

        # create a VTK rectilinear grid to sample onto
        structured_grid = vtk["mod"].vtkRectilinearGrid()
        structured_grid.SetDimensions(shape)
        structured_grid.SetXCoordinates(vtk["numpy_to_vtk"](x))
        structured_grid.SetYCoordinates(vtk["numpy_to_vtk"](y))
        structured_grid.SetZCoordinates(vtk["numpy_to_vtk"](z))

        # create and execute VTK interpolator
        interpolator = vtk["mod"].vtkResampleWithDataSet()
        interpolator.SetInputData(structured_grid)
        interpolator.SetSourceData(self._vtk_obj)
        interpolator.Update()
        interpolated = interpolator.GetOutput()

        # get results in a numpy representation
        array_id = 0 if self.values.name is None else self.values.name
        values_numpy = vtk["vtk_to_numpy"](interpolated.GetPointData().GetAbstractArray(array_id))

        # fill points without interpolated values
        if fill_value != 0:
            mask = vtk["vtk_to_numpy"](
                interpolated.GetPointData().GetAbstractArray("vtkValidPointMask")
            )
            values_numpy[mask != 1] = fill_value

        # VTK arrays are the z-y-x order, reorder interpolation results to x-y-z order
        values_reordered = np.transpose(np.reshape(values_numpy, shape[::-1]), (2, 1, 0))

        return values_reordered

    @abstractmethod
    def _interp_py(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        fill_value: float,
        max_samples_per_step: int,
        max_cells_per_step: int,
        rel_tol: float,
    ) -> ArrayLike:
        """Dimensionality-specific function (2D and 3D) to interpolate data at provided x, y, and z
        using vectorized python implementation.

        Parameters
        ----------
        x : Union[float, ArrayLike]
            x-coordinates of sampling points.
        y : Union[float, ArrayLike]
            y-coordinates of sampling points.
        z : Union[float, ArrayLike]
            z-coordinates of sampling points.
        fill_value : float
            Value to use when filling points without interpolated values.
        max_samples_per_step : int
            Max number of points to interpolate at per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        max_cells_per_step : int
            Max number of cells to interpolate from per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        rel_tol : float
            Relative tolerance when determining whether a point belongs to a cell.

        Returns
        -------
        ArrayLike
            Interpolated data.
        """

    def _interp_py_general(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        fill_value: float,
        max_samples_per_step: int,
        max_cells_per_step: int,
        rel_tol: float,
        axis_ignore: Union[Axis, None],
    ) -> ArrayLike:
        """A general function (2D and 3D) to interpolate data at provided x, y, and z using
        vectorized python implementation.

        Parameters
        ----------
        x : Union[float, ArrayLike]
            x-coordinates of sampling points.
        y : Union[float, ArrayLike]
            y-coordinates of sampling points.
        z : Union[float, ArrayLike]
            z-coordinates of sampling points.
        fill_value : float
            Value to use when filling points without interpolated values.
        max_samples_per_step : int
            Max number of points to interpolate at per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        max_cells_per_step : int
            Max number of cells to interpolate from per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        rel_tol : float
            Relative tolerance when determining whether a point belongs to a cell.
        axis_ignore : Union[Axis, None]
            When interpolating from a 2D dataset, must specify normal axis.

        Returns
        -------
        ArrayLike
            Interpolated data.
        """

        # get dimensionality of data
        num_dims = self._point_dims()

        if num_dims == 2 and axis_ignore is None:
            raise DataError("Must porvide 'axis_ignore' when interpolating from a 2d dataset.")

        xyz_grid = [x, y, z]

        if axis_ignore is not None:
            xyz_grid.pop(axis_ignore)

        # get numpy arrays for points and cells
        cell_connections = (
            self.cells.values
        )  # (num_cells, num_cell_vertices), num_cell_vertices=num_cell_faces
        points = self.points.values  # (num_points, num_dims)

        num_cells = len(cell_connections)
        num_points = len(points)

        # compute tolerances based on total size of unstructured grid
        bounds = self.bounds
        size = np.subtract(bounds[1], bounds[0])
        tol = size * rel_tol
        diag_tol = np.linalg.norm(tol)

        # compute (index) positions of unstructured points w.r.t. target Cartesian grid points
        # (i.e. between which Cartesian grid points a given unstructured grid point is located)
        # we perturb grid values in both directions to make sure we don't miss any points
        # due to numerical precision
        xyz_pos_l = np.zeros((num_dims, num_points), dtype=int)
        xyz_pos_r = np.zeros((num_dims, num_points), dtype=int)
        for dim in range(num_dims):
            xyz_pos_l[dim] = np.searchsorted(xyz_grid[dim] + tol[dim], points[:, dim])
            xyz_pos_r[dim] = np.searchsorted(xyz_grid[dim] - tol[dim], points[:, dim])

        # let's allocate an array for resulting values
        # every time we process a chunk of samples, we will write into this array
        interpolated_values = fill_value + np.zeros(
            [len(xyz_comp) for xyz_comp in xyz_grid], dtype=self.values.dtype
        )

        processed_cells_global = 0

        # to ovoid OOM for large datasets, we process only certain number of cells at a time
        while processed_cells_global < num_cells:
            target_processed_cells_global = min(
                num_cells, processed_cells_global + max_cells_per_step
            )

            connections_to_process = cell_connections[
                processed_cells_global:target_processed_cells_global
            ]

            # now we transfer this information to each cell. That is, each cell knows how its vertices
            # positioned relative to Cartesian grid points.
            # (num_dims, num_cells, num_vertices=num_cell_faces)
            xyz_pos_l_per_cell = xyz_pos_l[:, connections_to_process]
            xyz_pos_r_per_cell = xyz_pos_r[:, connections_to_process]

            # taking min/max among all cell vertices (per each dimension separately)
            # we get min and max indices of Cartesian grid points that may receive their values
            # from a given cell.
            # (num_dims, num_cells)
            cell_ind_min = np.min(xyz_pos_l_per_cell, axis=2)
            cell_ind_max = np.max(xyz_pos_r_per_cell, axis=2)

            # calculate number of Cartesian grid points where we will perform interpolation for a given
            # cell. Note that this number is much larger than actually needed, because essentially for
            # each cell we consider all Cartesian grid points that fall into the cell's bounding box.
            # We use word "sample" to represent such Cartesian grid points.
            # (num_cells,)
            num_samples_per_cell = np.prod(cell_ind_max - cell_ind_min, axis=0)

            # find cells that have non-zero number of samples
            # we use "ne" as a shortcut for "non empty"
            ne_cells = num_samples_per_cell > 0  # (num_cells,)
            num_ne_cells = np.sum(ne_cells)
            # indices of cells with non-zero number of samples in the original list of cells
            # (num_cells,)
            ne_cell_inds = np.arange(processed_cells_global, target_processed_cells_global)[
                ne_cells
            ]

            # restrict to non-empty cells only
            num_samples_per_ne_cell = num_samples_per_cell[ne_cells]
            cum_num_samples_per_ne_cell = np.cumsum(num_samples_per_ne_cell)

            ne_cell_ind_min = cell_ind_min[:, ne_cells]
            ne_cell_ind_max = cell_ind_max[:, ne_cells]

            # Next we need to perform actual interpolation at all sample points
            # this is computationally expensive operation and because we try to do everything
            # in the vectorized form, it can require a lot of memory, sometimes even causing OOM errors.
            # To avoid that, we impose restrictions on how many cells/samples can be processed at a time
            # effectivelly performing these operations in chunks.
            # Note that currently this is done sequentially, but could be relatively easy to parallelize

            # start counters of how many cells/samples have been processed
            processed_samples = 0
            processed_cells = 0

            while processed_cells < num_ne_cells:
                # how many cells we would like to process by the end of this step
                target_processed_cells = min(num_ne_cells, processed_cells + max_cells_per_step)

                # find how many cells we can processed based on number of allowed samples
                target_processed_samples = processed_samples + max_samples_per_step
                target_processed_cells_from_samples = (
                    np.searchsorted(cum_num_samples_per_ne_cell, target_processed_samples) + 1
                )

                # take min between the two
                target_processed_cells = min(
                    target_processed_cells, target_processed_cells_from_samples
                )

                # select cells and corresponding samples to process
                step_ne_cell_ind_min = ne_cell_ind_min[:, processed_cells:target_processed_cells]
                step_ne_cell_ind_max = ne_cell_ind_max[:, processed_cells:target_processed_cells]
                step_ne_cell_inds = ne_cell_inds[processed_cells:target_processed_cells]

                # process selected cells and points
                xyz_inds, interpolated = self._interp_py_chunk(
                    xyz_grid=xyz_grid,
                    cell_inds=step_ne_cell_inds,
                    cell_ind_min=step_ne_cell_ind_min,
                    cell_ind_max=step_ne_cell_ind_max,
                    sdf_tol=diag_tol,
                )

                if num_dims == 3:
                    interpolated_values[xyz_inds[0], xyz_inds[1], xyz_inds[2]] = interpolated
                else:
                    interpolated_values[xyz_inds[0], xyz_inds[1]] = interpolated

                processed_cells = target_processed_cells
                processed_samples = cum_num_samples_per_ne_cell[target_processed_cells - 1]

            processed_cells_global = target_processed_cells_global

        # in case of 2d grid broadcast results along normal direction assuming translational
        # invariance
        if num_dims == 2:
            orig_shape = [len(x), len(y), len(z)]
            flat_shape = orig_shape.copy()
            flat_shape[axis_ignore] = 1
            interpolated_values = np.reshape(interpolated_values, flat_shape)
            interpolated_values = np.broadcast_to(
                interpolated_values, (len(x), len(y), len(z))
            ).copy()

        return interpolated_values

    def _interp_py_chunk(
        self,
        xyz_grid: Tuple[ArrayLike[float], ...],
        cell_inds: ArrayLike[int],
        cell_ind_min: ArrayLike[int],
        cell_ind_max: ArrayLike[int],
        sdf_tol: float,
    ) -> Tuple[Tuple[ArrayLike, ...], ArrayLike]:
        """For each cell listed in ``cell_inds`` perform interpolation at a rectilinear subarray of
        xyz_grid given by a (3D) index span (cell_ind_min, cell_ind_max).

        Parameters
        ----------
        xyz_grid : Tuple[ArrayLike[float], ...]
            x, y, and z coordiantes defining rectilinear grid.
        cell_inds : ArrayLike[int]
            Indices of cells to perfrom interpolation from.
        cell_ind_min : ArrayLike[int]
            Starting x, y, and z indices of points for interpolation for each cell.
        cell_ind_max : ArrayLike[int]
            End x, y, and z indices of points for interpolation for each cell.
        sdf_tol : float
            Effective zero level set value, below which a point is considered to be inside a cell.

        Returns
        -------
        Tuple[Tuple[ArrayLike, ...], ArrayLike]
            x, y, and z indices of interpolated values and values themselves.
        """

        # get dimensionality of data
        num_dims = self._point_dims()
        num_cell_faces = self._cell_num_vertices()

        # get mesh info as numpy arrays
        points = self.points.values  # (num_points, num_dims)
        data_values = self.values.values  # (num_points,)
        cell_connections = self.cells.values[cell_inds]

        # compute number of samples to generate per cell
        num_samples_per_cell = np.prod(cell_ind_max - cell_ind_min, axis=0)

        # at this point we know how many samples we need to perform per each cell and we also
        # know span indices of these samples (in x, y, and z arrays)

        # we would like to perform all interpolations in a vectorized form, however, we have
        # a different number of interpolation samples for different cells. Thus, we need to
        # arange all samples in a linear way (flatten). Basically, we want to have data in this
        # form:
        # cell_ind | x_ind | y_ind | z_ind
        # --------------------------------
        #        0 |    23 |     5 |    11
        #        0 |    23 |     5 |    12
        #        0 |    23 |     6 |    11
        #        0 |    23 |     6 |    12
        #        1 |    41 |    11 |     0
        #        1 |    42 |    11 |     0
        #      ... |   ... |   ... |   ...

        # to do that we start with performing arange for each cell, but in vectorized way
        # this gives us something like this
        # [0, 1, 2, 3,   0, 1,   0, 1, 2, 3, 4, 5, 6,   ...]
        # |<-cell 0->|<-cell 1->|<-     cell 2    ->|<- ...

        num_cells = len(num_samples_per_cell)
        num_samples_cumul = num_samples_per_cell.cumsum()
        num_samples_total = num_samples_cumul[-1]

        # one big arange array
        inds_flat = np.arange(num_samples_total)
        # now subtract previous number of samples
        inds_flat[num_samples_per_cell[0] :] -= np.repeat(
            num_samples_cumul[:-1], num_samples_per_cell[1:]
        )

        # convert flat indices into 3d/2d indices as:
        # x_ind = [23, 23, 23, 23,   41, 41,      ...]
        # y_ind = [ 5,  5,  5,  5,    6,  6,      ...]
        # z_ind = [11, 12, 11, 12,    0,  0,      ...]
        #         |<-  cell 0  ->|<- cell 1 ->|<- ...
        num_samples_y = np.repeat(cell_ind_max[1] - cell_ind_min[1], num_samples_per_cell)

        # note: in 2d x, y correspond to (x, y, z).pop(normal_axis)
        if num_dims == 3:
            num_samples_z = np.repeat(cell_ind_max[2] - cell_ind_min[2], num_samples_per_cell)
            inds_flat, z_inds = np.divmod(inds_flat, num_samples_z)

        x_inds, y_inds = np.divmod(inds_flat, num_samples_y)

        start_inds = np.repeat(cell_ind_min, num_samples_per_cell, axis=1)
        x_inds = x_inds + start_inds[0]
        y_inds = y_inds + start_inds[1]
        if num_dims == 3:
            z_inds = z_inds + start_inds[2]

        # finally, we repeat cell indices corresponding number of times to obtain how
        # (x_ind, y_ind, z_ind) map to cell indices. So, now we have four arras:
        # x_ind    = [23, 23, 23, 23,   41, 41,      ...]
        # y_ind    = [ 5,  5,  5,  5,    6,  6,      ...]
        # z_ind    = [11, 12, 11, 12,    0,  0,      ...]
        # cell_map = [ 0,  0,  0,  0,    1,  1,      ...]
        #            |<-  cell 0  ->|<- cell 1 ->|<- ...
        step_cell_map = np.repeat(np.arange(num_cells), num_samples_per_cell)

        # let's put these arrays aside for a moment and perform the second preparatory step
        # specifically, for each face of each cell we will compute normal vector and distance
        # to the opposing cell vertex. This will allows us quickly calculate SDF of a cell at
        # each sample point as well as perform linear interpolation.

        # first, we collect coordinates of cell vertices into a single array
        # (num_cells, num_cell_vertices, num_dims)
        cell_vertices = np.float64(points[cell_connections, :])

        # array for resulting normals and distances
        normal = np.zeros((num_cell_faces, num_cells, num_dims))
        dist = np.zeros((num_cell_faces, num_cells))

        # loop face by face
        # note that by face_ind we denote both index of face in a cell and index of the opposing vertex
        for face_ind in range(num_cell_faces):
            # select vertices forming the given face
            face_pinds = list(np.arange(num_cell_faces))
            face_pinds.pop(face_ind)

            # calculate normal to the face
            # in 3D: cross product of two vectors lying in the face plane
            # in 2D: (-ty, tx) for a vector (tx, ty) along the face
            p0 = cell_vertices[:, face_pinds[0]]
            p01 = cell_vertices[:, face_pinds[1]] - p0
            p0Opp = cell_vertices[:, face_ind] - p0
            if num_dims == 3:
                p02 = cell_vertices[:, face_pinds[2]] - p0
                n = np.cross(p01, p02)
            else:
                n = np.roll(p01, 1, axis=1)
                n[:, 0] = -n[:, 0]
            n_norm = np.linalg.norm(n, axis=1)
            n = n / n_norm[:, None]

            # compute distance to the opposing vertex by taking a dot product between normal
            # and a vector connecting the opposing vertex and the face
            d = np.einsum("ij,ij->i", n, p0Opp)

            # obtained normal direction is arbitrary here. We will orient it such that it points
            # away from the triangle (and distance to the opposing vertex is negative).
            to_flip = d > 0
            d[to_flip] *= -1
            n[to_flip, :] *= -1

            # set distances in degenerate triangles to something positive to ignore later
            dist_zero = d == 0
            if any(dist_zero):
                d[dist_zero] = 1

            # record obtained info
            normal[face_ind] = n
            dist[face_ind] = d

        # now we all set up to proceed with actual interpolation at each sample point
        # the main idea here is that:
        # - we use `cell_map` to grab normals and distances
        #   of cells in which the given sample point is (potentially) located.
        # - use `x_ind, y_ind, z_ind` to find actual coordinates of a given sample point
        # - combine the above two to calculate cell SDF and interpolated value at a given sample
        #   point
        # - having cell SDF at the sample point actually tells us whether its inside the cell
        #   (keep value) or outside of it (discard interpolated value)

        # to perform SDF calculation and interpolation we will loop face by face and recording
        # their contributions. That is,
        # cell_sdf = max(face0_sdf, face1_sdf, ...)
        # interpolated_value = value0 * face0_sdf / dist0_sdf + ...
        # (because face0_sdf / dist0_sdf is linear shape function for vertex0)
        sdf = -inf * np.ones(num_samples_total)
        interpolated = np.zeros(num_samples_total, dtype=self._double_type)

        # coordinates of each sample point
        sample_xyz = np.zeros((num_samples_total, num_dims))
        sample_xyz[:, 0] = xyz_grid[0][x_inds]
        sample_xyz[:, 1] = xyz_grid[1][y_inds]
        if num_dims == 3:
            sample_xyz[:, 2] = xyz_grid[2][z_inds]

        # loop face by face
        for face_ind in range(num_cell_faces):
            # find a vector connecting sample point and face
            if face_ind == 0:
                vertex_ind = 1  # anythin other than 0
                vec = sample_xyz - cell_vertices[step_cell_map, vertex_ind, :]

            if face_ind == 1:  # since three faces share a point only do this once
                vertex_ind = 0  # it belongs to every face 1, 2, and 3
                vec = sample_xyz - cell_vertices[step_cell_map, 0, :]

            # compute distance from every sample point to the face of corresponding cell
            # using dot product
            tmp = normal[face_ind, step_cell_map, :] * vec
            d = np.sum(tmp, axis=1)

            # take max between distance to obtain the overall SDF of a cell
            sdf = np.maximum(sdf, d)

            # perform linear interpolation. Here we use the fact that when computing face SDF
            # at a given point and dividing it by the distance to the opposing vertex we get
            # a linear shape function for that vertex. So, we just need to multiply that by
            # the data value at that vertex to find its contribution into intepolated value.
            # (decomposed in an attempt to reduce memory consumption)
            tmp = self._double_type(data_values[cell_connections[step_cell_map, face_ind]])
            tmp *= d
            tmp /= dist[face_ind, step_cell_map]

            # ignore degenerate cells
            dist_zero = dist[face_ind, step_cell_map] > 0
            if any(dist_zero):
                sdf[dist_zero] = 10 * sdf_tol

            interpolated += tmp

        # The resulting array of interpolated values contain multiple candidate values for
        # every Cartesian point because bounding boxes of cells overlap.
        # Thus, we need to keep only those that come cell actually containing a given point.
        # This can be easily determined by the sign of the cell SDF sampled at a given point.
        valid_samples = sdf < sdf_tol

        interpolated_valid = interpolated[valid_samples]
        xyz_valid_inds = []
        xyz_valid_inds.append(x_inds[valid_samples])
        xyz_valid_inds.append(y_inds[valid_samples])
        if num_dims == 3:
            xyz_valid_inds.append(z_inds[valid_samples])

        return xyz_valid_inds, interpolated_valid

    @abstractmethod
    @requires_vtk
    def sel(
        self,
        x: Union[float, ArrayLike] = None,
        y: Union[float, ArrayLike] = None,
        z: Union[float, ArrayLike] = None,
    ) -> Union[UnstructuredGridDataset, SpatialDataArray]:
        """Extract/interpolate data along one or more Cartesian directions. At least of x, y, and z
        must be provided.

        Parameters
        ----------
        x : Union[float, ArrayLike] = None
            x-coordinate of the slice.
        y : Union[float, ArrayLike] = None
            y-coordinate of the slice.
        z : Union[float, ArrayLike] = None
            z-coordinate of the slice.

        Returns
        -------
        Union[TriangularGridDataset, SpatialDataArray]
            Extracted data.
        """

    @requires_vtk
    def sel_inside(self, bounds: Bound) -> UnstructuredGridDataset:
        """Return a new UnstructuredGridDataset that contains the minimal amount data necessary to
        cover a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        UnstructuredGridDataset
            Extracted spatial data array.
        """
        if any(bmin > bmax for bmin, bmax in zip(*bounds)):
            raise DataError(
                "Min and max bounds must be packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``."
            )

        data_bounds = self.bounds
        tol = 1e-6

        # For extracting cells covering target region we use vtk's filter that extract cells based
        # on provided implicit function. However, when we provide to it the implicit function of
        # the entire box, it has a couple of issues coming from the fact that the algorithm
        # eliminates every cells for which the implicit function has positive sign at all vertices.
        # As result, sometimes there are cells that despite overlaping with the target domain still
        # being eliminated. Two common cases:
        # - near corners of the target domain
        # - target domain is very thin
        # That's why we perform selection by sequentially eliminating cells on the outer side of
        # each of the 6 surfaces of the bounding box separately.
        tmp = self._vtk_obj
        for direction in range(2):
            for dim in range(3):
                sign = -1 + 2 * direction
                plane_pos = bounds[direction][dim]

                # Dealing with situation when target region does intersect with any cell:
                # in this case we shift target region so that it barely touches at least some
                # of cells
                if sign < 0 and plane_pos > data_bounds[1][dim] - tol:
                    plane_pos = data_bounds[1][dim] - tol
                if sign > 0 and plane_pos < data_bounds[0][dim] + tol:
                    plane_pos = data_bounds[0][dim] + tol

                # if all cells are on the inside side of the plane for a given surface
                # we don't need to check for intersection
                if plane_pos <= data_bounds[1][dim] and plane_pos >= data_bounds[0][dim]:
                    plane = vtk["mod"].vtkPlane()
                    center = [0, 0, 0]
                    normal = [0, 0, 0]
                    center[dim] = plane_pos
                    normal[dim] = sign
                    plane.SetOrigin(center)
                    plane.SetNormal(normal)
                    extractor = vtk["mod"].vtkExtractGeometry()
                    extractor.SetImplicitFunction(plane)
                    extractor.ExtractInsideOn()
                    extractor.ExtractBoundaryCellsOn()
                    extractor.SetInputData(tmp)
                    extractor.Update()
                    tmp = extractor.GetOutput()

        return self._from_vtk_obj(tmp, remove_degenerate_cells=True, remove_unused_points=True)

    def does_cover(self, bounds: Bound) -> bool:
        """Check whether data fully covers specified by ``bounds`` spatial region. If data contains
        only one point along a given direction, then it is assumed the data is constant along that
        direction and coverage is not checked.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        bool
            Full cover check outcome.
        """

        return all(
            (dmin <= smin and dmax >= smax)
            for dmin, dmax, smin, smax in zip(self.bounds[0], self.bounds[1], bounds[0], bounds[1])
        )

    @requires_vtk
    def reflect(
        self, axis: Axis, center: float, reflection_only: bool = False
    ) -> UnstructuredGridDataset:
        """Reflect unstructured data across the plane define by parameters ``axis`` and ``center``.
        By default the original data is preserved, setting ``reflection_only`` to ``True`` will
        produce only deflected data.

        Parameters
        ----------
        axis : Literal[0, 1, 2]
            Normal direction of the reflection plane.
        center : float
            Location of the reflection plane along its normal direction.
        reflection_only : bool = False
            Return only reflected data.

        Returns
        -------
        UnstructuredGridDataset
            Data after reflextion is performed.
        """

        reflector = vtk["mod"].vtkReflectionFilter()
        reflector.SetPlane([reflector.USE_X, reflector.USE_Y, reflector.USE_Z][axis])
        reflector.SetCenter(center)
        reflector.SetCopyInput(not reflection_only)
        reflector.SetInputData(self._vtk_obj)
        reflector.Update()

        return self._from_vtk_obj(reflector.GetOutput())


class TriangularGridDataset(UnstructuredGridDataset):
    """Dataset for storing triangular grid data. Data values are associated with the nodes of
    the grid.

    Note
    ----
    To use full functionality of unstructured datasets one must install ``vtk`` package (``pip
    install tidy3d[vtk]`` or ``pip install vtk``). Otherwise the functionality of unstructured
    datasets is limited to creation, writing to/loading from a file, and arithmetic manipulations.

    Example
    -------
    >>> tri_grid_points = PointDataArray(
    ...     [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    ...     coords=dict(index=np.arange(4), axis=np.arange(2)),
    ... )
    >>>
    >>> tri_grid_cells = CellDataArray(
    ...     [[0, 1, 2], [1, 2, 3]],
    ...     coords=dict(cell_index=np.arange(2), vertex_index=np.arange(3)),
    ... )
    >>>
    >>> tri_grid_values = IndexedDataArray(
    ...     [1.0, 2.0, 3.0, 4.0], coords=dict(index=np.arange(4)),
    ... )
    >>>
    >>> tri_grid = TriangularGridDataset(
    ...     normal_axis=1,
    ...     normal_pos=0,
    ...     points=tri_grid_points,
    ...     cells=tri_grid_cells,
    ...     values=tri_grid_values,
    ... )
    """

    normal_axis: Axis = pd.Field(
        ...,
        title="Grid Axis",
        description="Orientation of the grid.",
    )

    normal_pos: float = pd.Field(
        ...,
        title="Position",
        description="Coordinate of the grid along the normal direction.",
    )

    @cached_property
    def bounds(self) -> Bound:
        """Grid bounds."""
        bounds_2d = super().bounds
        bounds_3d = self._points_2d_to_3d(bounds_2d)
        return tuple(bounds_3d[0]), tuple(bounds_3d[1])

    @classmethod
    def _point_dims(cls) -> pd.PositiveInt:
        """Dimensionality of stored grid point coordinates."""
        return 2

    def _points_2d_to_3d(self, pts: ArrayLike) -> ArrayLike:
        """Convert 2d points into 3d points."""
        return np.insert(pts, obj=self.normal_axis, values=self.normal_pos, axis=1)

    @cached_property
    def _points_3d_array(self) -> ArrayLike:
        """3D representation of grid points."""
        return self._points_2d_to_3d(self.points.data)

    @classmethod
    def _cell_num_vertices(cls) -> pd.PositiveInt:
        """Number of vertices in a cell."""
        return 3

    @classmethod
    @requires_vtk
    def _vtk_cell_type(cls):
        """VTK cell type to use in the VTK representation."""
        return vtk["mod"].VTK_TRIANGLE

    @classmethod
    @requires_vtk
    def _from_vtk_obj(
        cls,
        vtk_obj,
        field=None,
        remove_degenerate_cells: bool = False,
        remove_unused_points: bool = False,
    ):
        """Initialize from a vtkUnstructuredGrid instance."""

        # get points cells data from vtk object
        if isinstance(vtk_obj, vtk["mod"].vtkPolyData):
            cells_vtk = vtk_obj.GetPolys()
        elif isinstance(vtk_obj, vtk["mod"].vtkUnstructuredGrid):
            cells_vtk = vtk_obj.GetCells()

        cells_numpy = vtk["vtk_to_numpy"](cells_vtk.GetConnectivityArray())

        cell_offsets = vtk["vtk_to_numpy"](cells_vtk.GetOffsetsArray())
        if not np.all(np.diff(cell_offsets) == cls._cell_num_vertices()):
            raise DataError(
                "Only triangular 'vtkUnstructuredGrid' or 'vtkPolyData' can be converted into "
                "'TriangularGridDataset'."
            )

        points_numpy = vtk["vtk_to_numpy"](vtk_obj.GetPoints().GetData())

        # data values are read directly into Tidy3D array
        values = cls._get_values_from_vtk(vtk_obj, len(points_numpy), field)

        # detect zero size dimension
        bounds = np.max(points_numpy, axis=0) - np.min(points_numpy, axis=0)
        zero_dims = np.where(np.isclose(bounds, 0))[0]

        if len(zero_dims) != 1:
            raise DataError(
                f"Provided vtk grid does not represent a two dimensional grid. Found zero size dimensions are {zero_dims}."
            )

        normal_axis = zero_dims[0]
        normal_pos = points_numpy[0][normal_axis]
        tan_dims = [0, 1, 2]
        tan_dims.remove(normal_axis)

        # convert 3d coordinates into 2d
        points_2d_numpy = points_numpy[:, tan_dims]

        # create Tidy3D points and cells arrays
        num_cells = len(cells_numpy) // cls._cell_num_vertices()
        cells_numpy = np.reshape(cells_numpy, (num_cells, cls._cell_num_vertices()))

        cells = CellDataArray(
            cells_numpy,
            coords=dict(
                cell_index=np.arange(num_cells), vertex_index=np.arange(cls._cell_num_vertices())
            ),
        )

        points = PointDataArray(
            points_2d_numpy,
            coords=dict(index=np.arange(len(points_numpy)), axis=np.arange(cls._point_dims())),
        )

        if remove_degenerate_cells:
            cells = cls._remove_degenerate_cells(cells=cells)

        if remove_unused_points:
            points, values, cells = cls._remove_unused_points(
                points=points, values=values, cells=cells
            )

        return cls(
            normal_axis=normal_axis,
            normal_pos=normal_pos,
            points=points,
            cells=cells,
            values=values,
        )

    @requires_vtk
    def plane_slice(self, axis: Axis, pos: float) -> SpatialDataArray:
        """Slice data with a plane and return the resulting line as a SpatialDataArray.

        Parameters
        ----------
        axis : Axis
            The normal direction of the slicing plane.
        pos : float
            Position of the slicing plane along its normal direction.

        Returns
        -------
        SpatialDataArray
            The resulting slice.
        """

        if axis == self.normal_axis:
            raise DataError(
                f"Triangular grid (normal: {self.normal_axis}) cannot be sliced by a parallel "
                "plane."
            )

        # perform slicing in vtk and get unprocessed points and values
        slice_vtk = self._plane_slice_raw(axis=axis, pos=pos)
        points_numpy = vtk["vtk_to_numpy"](slice_vtk.GetPoints().GetData())
        values = self._get_values_from_vtk(slice_vtk, len(points_numpy))

        # axis of the resulting line
        slice_axis = 3 - self.normal_axis - axis

        # sort found intersection in ascending order
        sorting = np.argsort(points_numpy[:, slice_axis], kind="mergesort")

        # assemble coords for SpatialDataArray
        coords = [None, None, None]
        coords[axis] = [pos]
        coords[self.normal_axis] = [self.normal_pos]
        coords[slice_axis] = points_numpy[sorting, slice_axis]
        coords_dict = dict(zip("xyz", coords))

        # reshape values from a 1d array into a 3d array
        new_shape = [1, 1, 1]
        new_shape[slice_axis] = len(values)
        values_reshaped = np.reshape(values.data[sorting], new_shape)

        return SpatialDataArray(values_reshaped, coords=coords_dict, name=self.values.name)

    @property
    def _triangulation_obj(self) -> Triangulation:
        """Matplotlib triangular representation of the grid to use in plotting."""
        return Triangulation(self.points[:, 0], self.points[:, 1], self.cells)

    @equal_aspect
    @add_ax_if_none
    def plot(
        self,
        ax: Ax = None,
        field: bool = True,
        grid: bool = True,
        cbar: bool = True,
        cmap: str = "viridis",
        vmin: float = None,
        vmax: float = None,
        shading: Literal["gourand", "flat"] = "gouraud",
        cbar_kwargs: Dict = None,
        pcolor_kwargs: Dict = None,
    ) -> Ax:
        """Plot the data field and/or the unstructured grid.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes = None
            matplotlib axes to plot on, if not specified, one is created.
        field : bool = True
            Whether to plot the data field.
        grid : bool = True
            Whether to plot the unstructured grid.
        cbar : bool = True
            Display colorbar (only if ``field == True``).
        cmap : str = "viridis"
            Color map to use for plotting.
        vmin : float = None
            The lower bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        vmax : float = None
            The upper bound of data range that the colormap covers. If ``None``, they are
            inferred from the data and other keyword arguments.
        shading : Literal["gourand", "flat"] = "gourand"
            Type of shading to use when plotting the data field.
        cbar_kwargs : Dict = {}
            Additional parameters passed to colorbar object.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        if cbar_kwargs is None:
            cbar_kwargs = {}
        if pcolor_kwargs is None:
            pcolor_kwargs = {}
        if not (field or grid):
            raise DataError("Nothing to plot ('field == False', 'grid == False').")

        # plot data field if requested
        if field:
            plot_obj = ax.tripcolor(
                self._triangulation_obj,
                self.values.data,
                shading=shading,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                **pcolor_kwargs,
            )

            if cbar:
                label_kwargs = {}
                if "label" not in cbar_kwargs:
                    label_kwargs["label"] = self.values.name
                plt.colorbar(plot_obj, **cbar_kwargs, **label_kwargs)

        # plot grid if requested
        if grid:
            ax.triplot(
                self._triangulation_obj,
                color=plot_params_grid.edgecolor,
                linewidth=plot_params_grid.linewidth,
            )

        # set labels and titles
        ax_labels = ["x", "y", "z"]
        normal_axis_name = ax_labels.pop(self.normal_axis)
        ax.set_xlabel(ax_labels[0])
        ax.set_ylabel(ax_labels[1])
        ax.set_title(f"{normal_axis_name} = {self.normal_pos}")
        return ax

    def interp(
        self,
        x: Union[float, ArrayLike],
        y: Union[float, ArrayLike],
        z: Union[float, ArrayLike],
        fill_value: Union[float, Literal["extrapolate"]] = None,
        use_vtk: bool = False,
        method: Literal["linear", "nearest"] = "linear",
        ignore_normal_pos: bool = True,
        max_samples_per_step: int = DEFAULT_MAX_SAMPLES_PER_STEP,
        max_cells_per_step: int = DEFAULT_MAX_CELLS_PER_STEP,
        rel_tol: float = DEFAULT_TOLERANCE_CELL_FINDING,
    ) -> SpatialDataArray:
        """Interpolate data at provided x, y, and z. Note that data is assumed to be invariant along
        the dataset's normal direction.

        Parameters
        ----------
        x : Union[float, ArrayLike]
            x-coordinates of sampling points.
        y : Union[float, ArrayLike]
            y-coordinates of sampling points.
        z : Union[float, ArrayLike]
            z-coordinates of sampling points.
        fill_value : Union[float, Literal["extrapolate"]] = 0
            Value to use when filling points without interpolated values. If ``"extrapolate"`` then
            nearest values are used. Note: in a future version the default value will be changed
            to ``"extrapolate"``.
        use_vtk : bool = False
            Use vtk's interpolation functionality or Tidy3D's own implementation. Note: this
            option will be removed in a future version.
        method: Literal["linear", "nearest"] = "linear"
            Interpolation method to use.
        ignore_normal_pos : bool = True
            (Depreciated) Assume data is invariant along the normal direction to the grid plane.
        max_samples_per_step : int = 1e4
            Max number of points to interpolate at per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        max_cells_per_step : int = 1e4
            Max number of cells to interpolate from per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        rel_tol : float = 1e-6
            Relative tolerance when determining whether a point belongs to a cell.

        Returns
        -------
        SpatialDataArray
            Interpolated data.
        """

        if fill_value is None:
            log.warning(
                "Default parameter setting 'fill_value=0' will be changed to "
                "'fill_value=``extrapolate``' in a future version."
            )
            fill_value = 0

        if not ignore_normal_pos:
            log.warning(
                "Parameter 'ignore_normal_pos' is depreciated. It is always assumed that data "
                "contained in 'TriangularGridDataset' is invariant in the normal direction. "
                "That is, 'ignore_normal_pos=True' is used."
            )

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        xyz = [x, y, z]
        xyz[self.normal_axis] = [self.normal_pos]
        interp_inplane = super().interp(
            **dict(zip("xyz", xyz)),
            fill_value=fill_value,
            use_vtk=use_vtk,
            method=method,
            max_samples_per_step=max_samples_per_step,
            max_cells_per_step=max_cells_per_step,
        )
        interp_broadcasted = np.broadcast_to(
            interp_inplane, [len(np.atleast_1d(comp)) for comp in [x, y, z]]
        )

        return SpatialDataArray(
            interp_broadcasted, coords=dict(x=x, y=y, z=z), name=self.values.name
        )

    def _interp_py(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        fill_value: float,
        max_samples_per_step: int,
        max_cells_per_step: int,
        rel_tol: float,
    ) -> ArrayLike:
        """2D-specific function to interpolate data at provided x, y, and z
        using vectorized python implementation.

        Parameters
        ----------
        x : Union[float, ArrayLike]
            x-coordinates of sampling points.
        y : Union[float, ArrayLike]
            y-coordinates of sampling points.
        z : Union[float, ArrayLike]
            z-coordinates of sampling points.
        fill_value : float
            Value to use when filling points without interpolated values.
        max_samples_per_step : int
            Max number of points to interpolate at per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        max_cells_per_step : int
            Max number of cells to interpolate from per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        rel_tol : float
            Relative tolerance when determining whether a point belongs to a cell.

        Returns
        -------
        ArrayLike
            Interpolated data.
        """

        return self._interp_py_general(
            x=x,
            y=y,
            z=z,
            fill_value=fill_value,
            max_samples_per_step=max_samples_per_step,
            max_cells_per_step=max_cells_per_step,
            rel_tol=rel_tol,
            axis_ignore=self.normal_axis,
        )

    @requires_vtk
    def sel(
        self,
        x: Union[float, ArrayLike] = None,
        y: Union[float, ArrayLike] = None,
        z: Union[float, ArrayLike] = None,
    ) -> SpatialDataArray:
        """Extract/interpolate data along one or more Cartesian directions. At least of x, y, and z
        must be provided.

        Parameters
        ----------
        x : Union[float, ArrayLike] = None
            x-coordinate of the slice.
        y : Union[float, ArrayLike] = None
            y-coordinate of the slice.
        z : Union[float, ArrayLike] = None
            z-coordinate of the slice.

        Returns
        -------
        SpatialDataArray
            Extracted data.
        """

        xyz = [x, y, z]
        axes = [ind for ind, comp in enumerate(xyz) if comp is not None]
        num_provided = len(axes)

        if self.normal_axis in axes:
            if xyz[self.normal_axis] != self.normal_pos:
                raise DataError(
                    f"No data for {'xyz'[self.normal_axis]} = {xyz[self.normal_axis]} (unstructured"
                    f" grid is defined at {'xyz'[self.normal_axis]} = {self.normal_pos})."
                )

            if num_provided < 3:
                num_provided -= 1
                axes.remove(self.normal_axis)

        if num_provided == 0:
            raise DataError("At least one of 'x', 'y', and 'z' must be specified.")

        if num_provided == 1:
            axis = axes[0]
            return self.plane_slice(axis=axis, pos=xyz[axis])

        if num_provided == 2:
            pos = [x, y, z]
            pos[self.normal_axis] = [self.normal_pos]
            return self.interp(x=pos[0], y=pos[1], z=pos[2])

        if num_provided == 3:
            return self.interp(x=x, y=y, z=z)

    @requires_vtk
    def reflect(
        self, axis: Axis, center: float, reflection_only: bool = False
    ) -> UnstructuredGridDataset:
        """Reflect unstructured data across the plane define by parameters ``axis`` and ``center``.
        By default the original data is preserved, setting ``reflection_only`` to ``True`` will
        produce only deflected data.

        Parameters
        ----------
        axis : Literal[0, 1, 2]
            Normal direction of the reflection plane.
        center : float
            Location of the reflection plane along its normal direction.
        reflection_only : bool = False
            Return only reflected data.

        Returns
        -------
        UnstructuredGridDataset
            Data after reflextion is performed.
        """

        # disallow reflecting along normal direction
        if axis == self.normal_axis:
            if reflection_only:
                return self.updated_copy(normal_pos=2 * center - self.normal_pos)
            else:
                raise DataError(
                    "Reflection in the normal direction to the grid is prohibited unless 'reflection_only=True'."
                )

        return super().reflect(axis=axis, center=center, reflection_only=reflection_only)

    @requires_vtk
    def sel_inside(self, bounds: Bound) -> TriangularGridDataset:
        """Return a new ``TriangularGridDataset`` that contains the minimal amount data necessary to
        cover a spatial region defined by ``bounds``.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        TriangularGridDataset
            Extracted spatial data array.
        """
        if any(bmin > bmax for bmin, bmax in zip(*bounds)):
            raise DataError(
                "Min and max bounds must be packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``."
            )

        # expand along normal direction
        new_bounds = [list(bounds[0]), list(bounds[1])]

        new_bounds[0][self.normal_axis] = -inf
        new_bounds[1][self.normal_axis] = inf

        return super().sel_inside(new_bounds)

    def does_cover(self, bounds: Bound) -> bool:
        """Check whether data fully covers specified by ``bounds`` spatial region. If data contains
        only one point along a given direction, then it is assumed the data is constant along that
        direction and coverage is not checked.


        Parameters
        ----------
        bounds : Tuple[float, float, float], Tuple[float, float float]
            Min and max bounds packaged as ``(minx, miny, minz), (maxx, maxy, maxz)``.

        Returns
        -------
        bool
            Full cover check outcome.
        """

        # expand along normal direction
        new_bounds = [list(bounds[0]), list(bounds[1])]

        new_bounds[0][self.normal_axis] = self.normal_pos
        new_bounds[1][self.normal_axis] = self.normal_pos

        return super().does_cover(new_bounds)


class TetrahedralGridDataset(UnstructuredGridDataset):
    """Dataset for storing tetrahedral grid data. Data values are associated with the nodes of
    the grid.

    Note
    ----
    To use full functionality of unstructured datasets one must install ``vtk`` package (``pip
    install tidy3d[vtk]`` or ``pip install vtk``). Otherwise the functionality of unstructured
    datasets is limited to creation, writing to/loading from a file, and arithmetic manipulations.

    Example
    -------
    >>> tet_grid_points = PointDataArray(
    ...     [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ...     coords=dict(index=np.arange(4), axis=np.arange(3)),
    ... )
    >>>
    >>> tet_grid_cells = CellDataArray(
    ...     [[0, 1, 2, 3]],
    ...     coords=dict(cell_index=np.arange(1), vertex_index=np.arange(4)),
    ... )
    >>>
    >>> tet_grid_values = IndexedDataArray(
    ...     [1.0, 2.0, 3.0, 4.0], coords=dict(index=np.arange(4)),
    ... )
    >>>
    >>> tet_grid = TetrahedralGridDataset(
    ...     points=tet_grid_points,
    ...     cells=tet_grid_cells,
    ...     values=tet_grid_values,
    ... )
    """

    @classmethod
    def _point_dims(cls) -> pd.PositiveInt:
        """Dimensionality of stored grid point coordinates."""
        return 3

    @cached_property
    def _points_3d_array(self) -> Bound:
        """3D coordinates of grid points."""
        return self.points.data

    @classmethod
    def _cell_num_vertices(cls) -> pd.PositiveInt:
        """Number of vertices in a cell."""
        return 4

    @classmethod
    @requires_vtk
    def _vtk_cell_type(cls):
        """VTK cell type to use in the VTK representation."""
        return vtk["mod"].VTK_TETRA

    @classmethod
    @requires_vtk
    def _from_vtk_obj(
        cls,
        grid,
        field=None,
        remove_degenerate_cells: bool = False,
        remove_unused_points: bool = False,
    ) -> TetrahedralGridDataset:
        """Initialize from a vtkUnstructuredGrid instance."""

        # read point, cells, and values info from a vtk instance
        cells_numpy = vtk["vtk_to_numpy"](grid.GetCells().GetConnectivityArray())
        points_numpy = vtk["vtk_to_numpy"](grid.GetPoints().GetData())
        values = cls._get_values_from_vtk(grid, len(points_numpy), field)

        # verify cell_types
        cells_types = vtk["vtk_to_numpy"](grid.GetCellTypesArray())
        if not np.all(cells_types == cls._vtk_cell_type()):
            raise DataError("Only tetrahedral 'vtkUnstructuredGrid' is currently supported")

        # pack point and cell information into Tidy3D arrays
        num_cells = len(cells_numpy) // cls._cell_num_vertices()
        cells_numpy = np.reshape(cells_numpy, (num_cells, cls._cell_num_vertices()))

        cells = CellDataArray(
            cells_numpy,
            coords=dict(
                cell_index=np.arange(num_cells), vertex_index=np.arange(cls._cell_num_vertices())
            ),
        )

        points = PointDataArray(
            points_numpy,
            coords=dict(index=np.arange(len(points_numpy)), axis=np.arange(cls._point_dims())),
        )

        if remove_degenerate_cells:
            cells = cls._remove_degenerate_cells(cells=cells)

        if remove_unused_points:
            points, values, cells = cls._remove_unused_points(
                points=points, values=values, cells=cells
            )

        return cls(points=points, cells=cells, values=values)

    @requires_vtk
    def plane_slice(self, axis: Axis, pos: float) -> TriangularGridDataset:
        """Slice data with a plane and return the resulting :class:.`TriangularGridDataset`.

        Parameters
        ----------
        axis : Axis
            The normal direction of the slicing plane.
        pos : float
            Position of the slicing plane along its normal direction.

        Returns
        -------
        TriangularGridDataset
            The resulting slice.
        """

        slice_vtk = self._plane_slice_raw(axis=axis, pos=pos)

        return TriangularGridDataset._from_vtk_obj(
            slice_vtk, remove_degenerate_cells=True, remove_unused_points=True
        )

    @requires_vtk
    def line_slice(self, axis: Axis, pos: Coordinate) -> SpatialDataArray:
        """Slice data with a line and return the resulting :class:.`SpatialDataArray`.

        Parameters
        ----------
        axis : Axis
            The axis of the slicing line.
        pos : Tuple[float, float, float]
            Position of the slicing line.

        Returns
        -------
        SpatialDataArray
            The resulting slice.
        """

        bounds = self.bounds
        start = list(pos)
        end = list(pos)

        start[axis] = bounds[0][axis]
        end[axis] = bounds[1][axis]

        # create cutting plane
        line = vtk["mod"].vtkLineSource()
        line.SetPoint1(start)
        line.SetPoint2(end)
        line.SetResolution(1)

        # this should be done using vtkProbeLineFilter
        # but for some reason it crashes Python
        # so, we use a workaround:
        # 1) extract cells that are intersected by line (to speed up further slicing)
        # 2) do plane slice along first direction
        # 3) do second plane slice along second direction

        prober = vtk["mod"].vtkExtractCellsAlongPolyLine()
        prober.SetSourceConnection(line.GetOutputPort())
        prober.SetInputData(self._vtk_obj)
        prober.Update()

        extracted_cells_vtk = prober.GetOutput()

        if extracted_cells_vtk.GetNumberOfPoints() == 0:
            raise DataError("Slicing line does not intersect the unstructured grid.")

        extracted_cells = TetrahedralGridDataset._from_vtk_obj(
            extracted_cells_vtk, remove_degenerate_cells=True, remove_unused_points=True
        )

        tan_dims = [0, 1, 2]
        tan_dims.remove(axis)

        # first plane slice
        plane_slice = extracted_cells.plane_slice(axis=tan_dims[0], pos=pos[tan_dims[0]])
        # second plane slice
        line_slice = plane_slice.plane_slice(axis=tan_dims[1], pos=pos[tan_dims[1]])

        return line_slice

    @requires_vtk
    def sel(
        self,
        x: Union[float, ArrayLike] = None,
        y: Union[float, ArrayLike] = None,
        z: Union[float, ArrayLike] = None,
    ) -> Union[TriangularGridDataset, SpatialDataArray]:
        """Extract/interpolate data along one or more Cartesian directions. At least of x, y, and z
        must be provided.

        Parameters
        ----------
        x : Union[float, ArrayLike] = None
            x-coordinate of the slice.
        y : Union[float, ArrayLike] = None
            y-coordinate of the slice.
        z : Union[float, ArrayLike] = None
            z-coordinate of the slice.

        Returns
        -------
        Union[TriangularGridDataset, SpatialDataArray]
            Extracted data.
        """

        xyz = [x, y, z]
        axes = [ind for ind, comp in enumerate(xyz) if comp is not None]

        num_provided = len(axes)

        if num_provided < 3 and any(not np.isscalar(comp) for comp in xyz if comp is not None):
            raise DataError(
                "Providing x, y, or z as array is only allowed for interpolation. That is, when all"
                " three x, y, and z are provided or method '.interp()' is used explicitly."
            )

        if num_provided == 0:
            raise DataError("At least one of 'x', 'y', and 'z' must be specified.")

        if num_provided == 1:
            axis = axes[0]
            return self.plane_slice(axis=axis, pos=xyz[axis])

        if num_provided == 2:
            axis = 3 - axes[0] - axes[1]
            xyz[axis] = 0
            return self.line_slice(axis=axis, pos=xyz)

        if num_provided == 3:
            return self.interp(x=x, y=y, z=z)

    def _interp_py(
        self,
        x: ArrayLike,
        y: ArrayLike,
        z: ArrayLike,
        fill_value: float,
        max_samples_per_step: int,
        max_cells_per_step: int,
        rel_tol: float,
    ) -> ArrayLike:
        """3D-specific function to interpolate data at provided x, y, and z
        using vectorized python implementation.

        Parameters
        ----------
        x : Union[float, ArrayLike]
            x-coordinates of sampling points.
        y : Union[float, ArrayLike]
            y-coordinates of sampling points.
        z : Union[float, ArrayLike]
            z-coordinates of sampling points.
        fill_value : float
            Value to use when filling points without interpolated values.
        max_samples_per_step : int
            Max number of points to interpolate at per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        max_cells_per_step : int
            Max number of cells to interpolate from per iteration (used only if `use_vtk=False`).
            Using a higher number may speed up calculations but, at the same time, it increases
            RAM usage.
        rel_tol : float
            Relative tolerance when determining whether a point belongs to a cell.

        Returns
        -------
        ArrayLike
            Interpolated data.
        """

        return self._interp_py_general(
            x=x,
            y=y,
            z=z,
            fill_value=fill_value,
            max_samples_per_step=max_samples_per_step,
            max_cells_per_step=max_cells_per_step,
            rel_tol=rel_tol,
            axis_ignore=None,
        )


UnstructuredGridDatasetType = Union[TriangularGridDataset, TetrahedralGridDataset]
CustomSpatialDataType = Union[SpatialDataArray, UnstructuredGridDatasetType]
CustomSpatialDataTypeAnnotated = Union[SpatialDataArray, annotate_type(UnstructuredGridDatasetType)]


def _get_numpy_array(data_array: Union[ArrayLike, DataArray, UnstructuredGridDataset]) -> ArrayLike:
    """Get numpy representation of dataarray/dataset values."""
    if isinstance(data_array, UnstructuredGridDataset):
        return data_array.values.values
    if isinstance(data_array, xr.DataArray):
        return data_array.values
    return np.array(data_array)


def _zeros_like(
    data_array: Union[ArrayLike, xr.DataArray, UnstructuredGridDataset],
) -> Union[ArrayLike, xr.DataArray, UnstructuredGridDataset]:
    """Get a zeroed replica of dataarray/dataset."""
    if isinstance(data_array, UnstructuredGridDataset):
        return data_array.updated_copy(values=xr.zeros_like(data_array.values))
    if isinstance(data_array, xr.DataArray):
        return xr.zeros_like(data_array)
    return np.zeros_like(data_array)


def _ones_like(
    data_array: Union[ArrayLike, xr.DataArray, UnstructuredGridDataset],
) -> Union[ArrayLike, xr.DataArray, UnstructuredGridDataset]:
    """Get a unity replica of dataarray/dataset."""
    if isinstance(data_array, UnstructuredGridDataset):
        return data_array.updated_copy(values=xr.ones_like(data_array.values))
    if isinstance(data_array, xr.DataArray):
        return xr.ones_like(data_array)
    return np.ones_like(data_array)


def _check_same_coordinates(
    a: Union[ArrayLike, xr.DataArray, UnstructuredGridDataset],
    b: Union[ArrayLike, xr.DataArray, UnstructuredGridDataset],
) -> bool:
    """Check whether two array are defined at the same coordinates."""

    # we can have xarray.DataArray's of different types but still same coordinates
    # we will deal with that case separately
    both_xarrays = isinstance(a, xr.DataArray) and isinstance(b, xr.DataArray)
    if (not both_xarrays) and type(a) is not type(b):
        return False

    if isinstance(a, UnstructuredGridDataset):
        if not np.allclose(a.points, b.points) or not np.all(a.cells == b.cells):
            return False

        if isinstance(a, TriangularGridDataset):
            if a.normal_axis != b.normal_axis or a.normal_pos != b.normal_pos:
                return False

    elif isinstance(a, xr.DataArray):
        if a.coords.keys() != b.coords.keys() or a.coords != b.coords:
            return False

    else:
        if np.shape(a) != np.shape(b):
            return False

    return True

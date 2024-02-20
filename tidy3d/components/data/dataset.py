"""Collections of DataArrays."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Dict, Callable, Any

import xarray as xr
import numpy as np
import pydantic.v1 as pd
from matplotlib.tri import Triangulation
from matplotlib import pyplot as plt
import numbers

from .data_array import DataArray
from .data_array import ScalarFieldDataArray, ScalarFieldTimeDataArray, ScalarModeFieldDataArray
from .data_array import ModeIndexDataArray, GroupIndexDataArray, ModeDispersionDataArray
from .data_array import TriangleMeshDataArray
from .data_array import TimeDataArray
from .data_array import PointDataArray, IndexedDataArray, CellDataArray, SpatialDataArray

from ..viz import equal_aspect, add_ax_if_none, plot_params_grid
from ..base import Tidy3dBaseModel, cached_property
from ..base import skip_if_fields_missing
from ..types import Axis, Bound, ArrayLike, Ax, Coordinate, Literal
from ...packaging import vtk, requires_vtk
from ...exceptions import DataError, ValidationError, Tidy3dNotImplementedError
from ...constants import PICOSECOND_PER_NANOMETER_PER_KILOMETER
from ...log import log


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

    @pd.validator("cells", always=True)
    def match_cells_to_vtk_type(cls, val):
        """Check that cell connections does not have duplicate points."""
        if vtk is None:
            return val

        # using val.astype(np.int32/64) directly causes issues when dataarray are later checked ==
        return CellDataArray(val.data.astype(vtk["id_type"], copy=False), coords=val.coords)

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

    @pd.validator("points", always=True)
    def points_right_dims(cls, val):
        """Check that point coordinates have the right dimensionality."""
        axis_coords_expected = np.arange(cls._point_dims())
        axis_coords_given = val.axis.data
        if np.any(axis_coords_given != axis_coords_expected):
            raise ValidationError(
                f"Points array is expected to have {axis_coords_expected} coord values along 'axis'"
                f" (given: {axis_coords_given})."
            )
        return val

    @pd.validator("cells", always=True)
    def cells_right_type(cls, val):
        """Check that cell are of the right type."""
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
        min_index_used = np.min(all_point_indices_used)
        max_index_used = np.max(all_point_indices_used)

        points = values.get("points")
        num_points = len(points)

        if max_index_used != num_points - 1 or min_index_used != 0:
            raise ValidationError(
                "Cell connections array uses undefined point indices in the range "
                f"[{min_index_used}, {max_index_used}]. The valid range of point indices is "
                f"[0, {num_points-1}]."
            )
        return val

    @pd.validator("cells", always=True)
    def check_valid_cells(cls, val):
        """Check that cell connections does not have duplicate points."""
        indices = val.data
        for i in range(cls._cell_num_vertices() - 1):
            for j in range(i + 1, cls._cell_num_vertices()):
                if np.any(indices[:, i] == indices[:, j]):
                    log.warning("Unstructured grid contains degenerate cells.")
        return val

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
    def _points_3d_array(self) -> Bound:
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
        point_data_vtk = vtk["numpy_to_vtk"](self.values.data)
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
    def _from_vtk_obj(cls, vtk_obj, field=None) -> UnstructuredGridDataset:
        """Initialize from a vtk object."""

    @classmethod
    @requires_vtk
    def from_vtu(cls, file: str, field: str = None) -> UnstructuredGridDataset:
        """Load unstructured data from a vtu file.

        Parameters
        ----------
        fname : str
            Full path to the .vtu file to load the unstructured data from.
        field : str = None
            Name of the field to load.

        Returns
        -------
        UnstructuredGridDataset
            Unstructured data.
        """
        grid = cls._read_vtkUnstructuredGrid(file)
        return cls._from_vtk_obj(grid, field=field)

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

            # currently we assume data is scalar
            num_components = array_vtk.GetNumberOfComponents()
            if num_components > 1:
                raise DataError(
                    f"Found point data array in a VTK object is expected to have only 1 component. Found {num_components} components."
                )

            # check that number of values matches number of grid points
            num_tuples = array_vtk.GetNumberOfTuples()
            if num_tuples != num_points:
                raise DataError(
                    f"The length of found point data array ({num_tuples}) does not match the number of grid points ({num_points})."
                )

            values_numpy = vtk["vtk_to_numpy"](array_vtk)
            values_name = array_vtk.GetName()

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

        return self._from_vtk_obj(clean_clip)

    @requires_vtk
    def interp(
        self,
        x: Union[float, ArrayLike],
        y: Union[float, ArrayLike],
        z: Union[float, ArrayLike],
        fill_value: float = 0,
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
        fill_value : float = 0
            Value to use when filling points without interpolated values.

        Returns
        -------
        SpatialDataArray
            Interpolated data.
        """

        # calculate the resulting array shape
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)
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

        return SpatialDataArray(values_reordered, coords=dict(x=x, y=y, z=z), name=self.values.name)

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
    """Dataset for storing triangular grid data.

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
    def _from_vtk_obj(cls, vtk_obj, field=None):
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

    @requires_vtk
    def interp(
        self,
        x: Union[float, ArrayLike],
        y: Union[float, ArrayLike],
        z: Union[float, ArrayLike],
        fill_value: float = 0,
        ignore_normal_pos: bool = True,
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
        fill_value : float = 0
            Value to use when filling points without interpolated values.
        ignore_normal_pos : bool = True
            Assume data is invariant along the normal direction to the grid plane.

        Returns
        -------
        SpatialDataArray
            Interpolated data.
        """

        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        z = np.atleast_1d(z)

        if ignore_normal_pos:
            xyz = [x, y, z]
            xyz[self.normal_axis] = [self.normal_pos]
            interp_inplane = super().interp(**dict(zip("xyz", xyz)), fill_value=fill_value)
            interp_broadcasted = np.broadcast_to(
                interp_inplane, [len(np.atleast_1d(comp)) for comp in [x, y, z]]
            )

            return SpatialDataArray(
                interp_broadcasted, coords=dict(x=x, y=y, z=z), name=self.values.name
            )

        return super().interp(x=x, y=y, z=z, fill_value=fill_value)

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
            raise DataError("Reflection in the normal direction to the grid is prohibited.")

        return super().reflect(axis=axis, center=center, reflection_only=reflection_only)


class TetrahedralGridDataset(UnstructuredGridDataset):
    """Dataset for storing tetrahedral grid data.

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
    def _from_vtk_obj(cls, grid, field=None) -> TetrahedralGridDataset:
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

        return TriangularGridDataset._from_vtk_obj(slice_vtk)

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

        extracted_cells = TetrahedralGridDataset._from_vtk_obj(extracted_cells_vtk)

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


UnstructuredGridDatasetType = Union[TriangularGridDataset, TetrahedralGridDataset]

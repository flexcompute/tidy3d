"""Defines abstract base for unstructured datasets."""

from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from typing import Literal, Tuple, Union

import numpy as np
import pydantic.v1 as pd
from xarray import DataArray as XrDataArray

from tidy3d.components.base import cached_property, skip_if_fields_missing
from tidy3d.components.data.data_array import (
    DATA_ARRAY_MAP,
    CellDataArray,
    IndexedDataArray,
    IndexedDataArrayTypes,
    PointDataArray,
    SpatialDataArray,
)
from tidy3d.components.data.dataset import Dataset
from tidy3d.components.types import ArrayLike, Axis, Bound
from tidy3d.constants import inf
from tidy3d.exceptions import DataError, Tidy3dNotImplementedError, ValidationError
from tidy3d.log import log
from tidy3d.packaging import requires_vtk, vtk

DEFAULT_MAX_SAMPLES_PER_STEP = 10_000
DEFAULT_MAX_CELLS_PER_STEP = 10_000
DEFAULT_TOLERANCE_CELL_FINDING = 1e-6


class UnstructuredGridDataset(Dataset, np.lib.mixins.NDArrayOperatorsMixin, ABC):
    """Abstract base for datasets that store unstructured grid data."""

    points: PointDataArray = pd.Field(
        ...,
        title="Grid Points",
        description="Coordinates of points composing the unstructured grid.",
    )

    values: IndexedDataArrayTypes = pd.Field(
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

    """ Fundametal parameters to set up based on grid dimensionality """

    @classmethod
    @abstractmethod
    def _point_dims(cls) -> pd.PositiveInt:
        """Dimensionality of stored grid point coordinates."""

    @classmethod
    @abstractmethod
    def _cell_num_vertices(cls) -> pd.PositiveInt:
        """Number of vertices in a cell."""

    """ Validators """

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
    def first_values_dim_is_index(cls, val):
        """Check that the number of data values matches the number of grid points."""
        if val.dims[0] != "index":
            raise ValidationError("First dimension of array 'values' must be 'index'.")
        return val

    @pd.validator("values", always=True)
    def values_right_indexing(cls, val):
        """Check that data values are indexed correctly."""
        # currently support only simple ordered indexing of points, that is, 0, 1, 2, ...
        indices_expected = np.arange(len(val.index.data))
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
        num_values = len(val.index)

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
        point_indices = set(np.arange(len(values["points"].data)))
        used_indices = set(values["cells"].values.ravel())

        if not point_indices.issubset(used_indices):
            log.warning(
                "Unstructured grid dataset contains unused points. "
                "Consider calling 'clean()' to remove them."
            )

        return values

    """ Convenience properties """

    @property
    def name(self) -> str:
        """Dataset name."""
        # we redirect name to values.name
        return self.values.name

    def rename(self, name: str) -> UnstructuredGridDataset:
        """Return a renamed array."""
        return self.updated_copy(values=self.values.rename(name))

    @property
    def is_complex(self) -> bool:
        """Data type."""
        return np.iscomplexobj(self.values)

    @property
    def _double_type(self):
        """Corresponding double data type."""
        return np.complex128 if self.is_complex else np.float64

    @property
    def is_uniform(self):
        """Whether each element is of equal value in ``values``."""
        return self.values.is_uniform

    @cached_property
    def _values_coords_dict(self):
        """Non-spatial dimensions are corresponding coordinate values of stored data."""
        coord_dict = {dim: self.values.coords[dim].data for dim in self.values.dims}
        _ = coord_dict.pop("index")
        return coord_dict

    @cached_property
    def _fields_shape(self):
        """Shape in which fields are stored."""
        return [len(coord) for coord in self._values_coords_dict.values()]

    @cached_property
    def _num_fields(self):
        """Total number of stored fields."""
        return 1 if len(self._fields_shape) == 0 else np.prod(self._fields_shape)

    @cached_property
    def _values_type(self):
        """Type of array storing values."""
        return type(self.values)

    @cached_property
    def bounds(self) -> Bound:
        """Grid bounds."""
        return tuple(np.min(self.points.data, axis=0)), tuple(np.max(self.points.data, axis=0))

    @cached_property
    @abstractmethod
    def _points_3d_array(self):
        """3D coordinates of grid points."""

    """ Grid cleaning """

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
        cls, points: PointDataArray, values: IndexedDataArrayTypes, cells: CellDataArray
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
            values = values.sel(index=used_indices)
            if "index" in values.coords:
                # renumber if index given as a coordinate
                values["index"] = np.arange(len(used_indices))

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

    """ Arithmetic operations """

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """Override of numpy functions."""

        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with a scalar or an unstructured grid dataset of the same spatial dimensionality
            if not (
                isinstance(x, numbers.Number)
                or (
                    isinstance(x, UnstructuredGridDataset) and x._point_dims() == self._point_dims()
                )
            ):
                raise Tidy3dNotImplementedError(
                    f"Cannot perform arithmetic operations between instances of different classes ({type(self)} and {type(x)})."
                )

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

    """ VTK interfacing """

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

        if len(self._fields_shape) > 0:
            data_values = data_values.reshape(
                (len(self.points.values), (1 + self.is_complex) * self._num_fields)
            )

        point_data_vtk = vtk["numpy_to_vtk"](data_values)
        point_data_vtk.SetName(self.name)
        grid.GetPointData().AddArray(point_data_vtk)

        return grid

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
        values_type=IndexedDataArray,
        expect_complex=None,
    ) -> UnstructuredGridDataset:
        """Initialize from a vtk object."""

    @requires_vtk
    def _from_vtk_obj_internal(
        self,
        vtk_obj,
        remove_degenerate_cells: bool = True,
        remove_unused_points: bool = True,
    ) -> UnstructuredGridDataset:
        """Initialize from a vtk object when performing internal operations. When we do that we
        pass structure of possibly multidimensional nature of values through parametes field and
        values_type. We also turn on by default cleaning of geometry."""
        return self._from_vtk_obj(
            vtk_obj=vtk_obj,
            field=self._values_coords_dict,
            remove_degenerate_cells=remove_degenerate_cells,
            remove_unused_points=remove_unused_points,
            values_type=self._values_type,
            expect_complex=self.is_complex,
        )

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
        cls,
        vtk_obj,
        num_points: pd.PositiveInt,
        field: str = None,
        values_type=IndexedDataArray,
        expect_complex=None,
    ) -> IndexedDataArray:
        """Get point data values from a VTK object."""

        point_data = vtk_obj.GetPointData()
        num_point_arrays = point_data.GetNumberOfArrays()

        if num_point_arrays == 0:
            log.warning(
                "No point data is found in a VTK object. '.values' will be initialized to zeros."
            )
            values_numpy = np.zeros(num_points)
            values_coords = {"index": []}
            values_name = None

        else:
            field_ind = field if isinstance(field, str) else 0

            array_vtk = point_data.GetAbstractArray(field_ind)
            # currently we assume data is real or complex scalar
            num_components = array_vtk.GetNumberOfComponents()
            if num_components > 2 and not isinstance(field, dict):
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
            if (num_components == 2 and expect_complex is None) or expect_complex is True:
                values_numpy = values_numpy.view("complex")

            new_shape = [num_points]
            if isinstance(field, dict):
                new_shape = new_shape + [len(coord) for coord in field.values()]

            values_numpy = np.reshape(values_numpy, new_shape)

            # currently we assume there is only one point data array provided in the VTK object
            if num_point_arrays > 1 and field is None:
                log.warning(
                    f"{num_point_arrays} point data arrays are found in a VTK object. "
                    f"Only the first array (name: {values_name}) will be used to initialize "
                    "'.values' while the rest will be ignored."
                )

            values_coords = dict(index=np.arange(num_points))
            if isinstance(field, dict):
                values_coords.update(field)

        values = values_type(values_numpy, coords=values_coords, name=values_name)

        return values

    def get_cell_values(self, **kwargs):
        """This function returns the cell values for the fields stored in the UnstructuredGridDataset.
        If multiple fields are stored per point, like in an IndexedVoltageDataArray, cell values
        will be provided for each of the fields unless a selection argument is provided, e.g., voltage=0.2
        """

        values = self.values.sel(**kwargs)

        return values[self.cells].mean(dim="vertex_index").values

    @abstractmethod
    def get_cell_volumes(self):
        """Get the volumes associated to each cell."""

    """ Grid operations """

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
    def plane_slice(self, axis: Axis, pos: float) -> Union[XrDataArray, UnstructuredGridDataset]:
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
        Union[xarray.DataArray, UnstructuredGridDataset]
            The resulting slice.
        """

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

        return self._from_vtk_obj_internal(clean_clip)

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

        # since reflection does not really change geometries, let's not clean it
        return self._from_vtk_obj_internal(
            reflector.GetOutput(), remove_degenerate_cells=False, remove_unused_points=False
        )

    """ Interpolation """

    def interp(
        self,
        x: Union[float, ArrayLike] = None,
        y: Union[float, ArrayLike] = None,
        z: Union[float, ArrayLike] = None,
        fill_value: Union[
            float, Literal["extrapolate"]
        ] = None,  # TODO: an array if multiple fields?
        use_vtk: bool = False,
        method: Literal["linear", "nearest"] = "linear",
        max_samples_per_step: int = DEFAULT_MAX_SAMPLES_PER_STEP,
        max_cells_per_step: int = DEFAULT_MAX_CELLS_PER_STEP,
        rel_tol: float = DEFAULT_TOLERANCE_CELL_FINDING,
        **coords_kwargs,
    ) -> XrDataArray:
        """Interpolate data along spatial dimensions x, y, and z and/or non-spatial dimensions.
        For spatial sampling points must provide all x, y, and z.

        Parameters
        ----------
        x : Union[float, ArrayLike] = None
            x-coordinates of sampling points.
        y : Union[float, ArrayLike] = None
            y-coordinates of sampling points.
        z : Union[float, ArrayLike] = None
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
        **coords_kwargs : dict
            Keyword arguments to pass to the xarray interp() function.

        Returns
        -------
        xarray.DataArray
            Interpolated data.
        """

        if fill_value is None:
            log.warning(
                "Default parameter setting 'fill_value=0' will be changed to "
                "'fill_value=``extrapolate``' in a future version."
            )
            fill_value = 0

        spatial_dims_given = any(comp is not None for comp in [x, y, z])
        if spatial_dims_given and any(comp is None for comp in [x, y, z]):
            raise DataError("Must provide either all or none of 'x', 'y', and 'z'")

        if not spatial_dims_given and len(coords_kwargs) == 0:
            raise DataError(
                "Must provide either 'x', 'y', and 'z' or points along other non-spatial dimensions."
            )

        result = self
        if len(coords_kwargs) > 0:
            result = result._non_spatial_interp(
                method=method, fill_value=fill_value, **coords_kwargs
            )

        if spatial_dims_given:
            result = result._spatial_interp(
                x=x,
                y=y,
                z=z,
                fill_value=fill_value,
                use_vtk=use_vtk,
                method=method,
                max_samples_per_step=max_samples_per_step,
                max_cells_per_step=max_cells_per_step,
                rel_tol=rel_tol,
            )

        return result

    def _non_spatial_interp(self, method="linear", fill_value=np.nan, **coords_kwargs):
        """Interpolate data at non-spatial dimensions using xarray's interp() function.

        Parameters
        ----------
        method: Literal["linear", "nearest"] = "linear"
            Interpolation method to use.
        fill_value : Union[float, Literal["extrapolate"]] = 0
            Value to use when filling points without interpolated values. If ``"extrapolate"`` then
            nearest values are used. Note: in a future version the default value will be changed
            to ``"extrapolate"``.
        **coords_kwargs : dict
            Keyword arguments to pass to the xarray interp() function.

        Returns
        -------
        xarray.DataArray
            Interpolated data.
        """
        coords_kwargs_only_lists = {
            key: value if isinstance(value, list) else [value]
            for key, value in coords_kwargs.items()
        }
        return self.updated_copy(
            values=self.values.interp(
                **coords_kwargs_only_lists,
                method="linear",
                kwargs=dict(fill_value=fill_value),
            )
        )

    def _spatial_interp(
        self,
        x: Union[float, ArrayLike],
        y: Union[float, ArrayLike],
        z: Union[float, ArrayLike],
        fill_value: Union[
            float, Literal["extrapolate"]
        ] = None,  # TODO: an array if multiple fields?
        use_vtk: bool = False,
        method: Literal["linear", "nearest"] = "linear",
        max_samples_per_step: int = DEFAULT_MAX_SAMPLES_PER_STEP,
        max_cells_per_step: int = DEFAULT_MAX_CELLS_PER_STEP,
        rel_tol: float = DEFAULT_TOLERANCE_CELL_FINDING,
    ) -> XrDataArray:
        """Interpolate data along spatial dimensions at provided x, y, and z.

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
        xarray.DataArray
            Interpolated data.
        """

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
                if self.is_complex:
                    raise DataError("Option 'use_vtk=True' is not supported for complex datasets.")
                if len(self._fields_shape) > 0:
                    raise DataError(
                        "Option 'use_vtk=True' is not supported for multidimensional datasets."
                    )
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

        coords_dict = dict(x=x, y=y, z=z)
        coords_dict.update(self._values_coords_dict)

        if len(self._values_coords_dict) == 0:
            return SpatialDataArray(interpolated_values, coords=coords_dict, name=self.values.name)
        else:
            return XrDataArray(interpolated_values, coords=coords_dict, name=self.values.name)

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
        # do a quick and dirty in case of multiple fields: just look at the very first field
        nans = np.isnan(values).reshape((len(x), len(y), len(z), self._num_fields))[:, :, :, 0]

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
        fill_value: float,  # TODO: an array if multidimensional
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

        # TODO: generalize this
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
            raise DataError("Must provide 'axis_ignore' when interpolating from a 2d dataset.")

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
            [len(xyz_comp) for xyz_comp in xyz_grid] + self._fields_shape, dtype=self.values.dtype
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
            orig_shape = [len(x), len(y), len(z)] + self._fields_shape
            flat_shape = orig_shape.copy()
            flat_shape[axis_ignore] = 1
            interpolated_values = np.reshape(interpolated_values, flat_shape)
            interpolated_values = np.broadcast_to(interpolated_values, orig_shape).copy()

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
        data_values = self.values  # (num_points,)
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
        interpolated = np.zeros([num_samples_total] + self._fields_shape, dtype=self._double_type)

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
            tmp = self._double_type(
                data_values.sel(index=cell_connections[step_cell_map, face_ind]).data
            )
            tmp *= np.reshape(d, [num_samples_total] + [1] * len(self._fields_shape))
            tmp /= np.reshape(
                dist[face_ind, step_cell_map], [num_samples_total] + [1] * len(self._fields_shape)
            )

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

    """ Data selection """

    @requires_vtk
    def sel(
        self,
        x: Union[float, ArrayLike] = None,
        y: Union[float, ArrayLike] = None,
        z: Union[float, ArrayLike] = None,
        method: Literal["None", "nearest", "pad", "ffill", "backfill", "bfill"] = None,
        **sel_kwargs,
    ) -> Union[UnstructuredGridDataset, XrDataArray]:
        """Extract/interpolate data along one or more spatial or non-spatial directions. Must provide at least one argument
        among 'x', 'y', 'z' or non-spatial dimensions through additional arguments. Along spatial dimensions a suitable slicing of
        grid is applied (plane slice, line slice, or interpolation). Selection along non-spatial dimensions is forwarded to
        .sel() xarray function. Parameter 'method' applies only to non-spatial dimensions.

        Parameters
        ----------
        x : Union[float, ArrayLike] = None
            x-coordinate of the slice.
        y : Union[float, ArrayLike] = None
            y-coordinate of the slice.
        z : Union[float, ArrayLike] = None
            z-coordinate of the slice.
        method: Literal[None, "nearest", "pad", "ffill", "backfill", "bfill"] = None
            Method to use in xarray sel() function.
        **sel_kwargs : dict
            Keyword arguments to pass to the xarray sel() function.

        Returns
        -------
        Union[TriangularGridDataset, xarray.DataArray]
            Extracted data.
        """

    def _non_spatial_sel(
        self,
        method=None,
        **sel_kwargs,
    ) -> XrDataArray:
        """Select/interpolate data along one or more non-Cartesian directions.

        Parameters
        ----------
        **sel_kwargs : dict
            Keyword arguments to pass to the xarray sel() function.

        Returns
        -------
        xarray.DataArray
            Extracted data.
        """

        if "index" in sel_kwargs.keys():
            raise DataError("Cannot select along dimension 'index'.")

        # convert individual values into lists of length 1
        # so that xarray doesn't drop the corresponding dimension
        sel_kwargs_only_lists = {
            key: value if isinstance(value, list) else [value] for key, value in sel_kwargs.items()
        }
        return self.updated_copy(values=self.values.sel(**sel_kwargs_only_lists, method=method))

    def isel(
        self,
        **sel_kwargs,
    ) -> XrDataArray:
        """Select data along one or more non-Cartesian directions by coordinate index.

        Parameters
        ----------
        **sel_kwargs : dict
            Keyword arguments to pass to the xarray isel() function.

        Returns
        -------
        xarray.DataArray
            Extracted data.
        """

        if "index" in sel_kwargs.keys():
            raise DataError("Cannot select along dimension 'index'.")

        # convert individual values into lists of length 1
        # so that xarray doesn't drop the corresponding dimension
        sel_kwargs_only_lists = {
            key: value if isinstance(value, list) else [value] for key, value in sel_kwargs.items()
        }
        return self.updated_copy(values=self.values.isel(**sel_kwargs_only_lists))

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

        return self._from_vtk_obj_internal(tmp)

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

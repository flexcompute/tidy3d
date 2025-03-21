"""Defines the FDTD grid."""

from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np
import pydantic.v1 as pd

from ...exceptions import SetupError
from ..base import Tidy3dBaseModel, cached_property
from ..data.data_array import DataArray, ScalarFieldDataArray, SpatialDataArray
from ..data.utils import UnstructuredGridDataset, UnstructuredGridDatasetType
from ..geometry.base import Box
from ..types import ArrayFloat1D, Axis, Coordinate, InterpMethod, Literal

# data type of one dimensional coordinate array.
Coords1D = ArrayFloat1D


class Coords(Tidy3dBaseModel):
    """Holds data about a set of x,y,z positions on a grid.

    Example
    -------
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-1, 1, 11)
    >>> z = np.linspace(-1, 1, 12)
    >>> coords = Coords(x=x, y=y, z=z)
    """

    x: Coords1D = pd.Field(
        ..., title="X Coordinates", description="1-dimensional array of x coordinates."
    )

    y: Coords1D = pd.Field(
        ..., title="Y Coordinates", description="1-dimensional array of y coordinates."
    )

    z: Coords1D = pd.Field(
        ..., title="Z Coordinates", description="1-dimensional array of z coordinates."
    )

    @property
    def to_dict(self):
        """Return a dict of the three Coord1D objects as numpy arrays."""
        return {key: self.dict()[key] for key in "xyz"}

    @property
    def to_list(self):
        """Return a list of the three Coord1D objects as numpy arrays."""
        return list(self.to_dict.values())

    @cached_property
    def cell_sizes(self) -> SpatialDataArray:
        """Returns the sizes of the cells in each coordinate array as a dictionary."""
        cell_sizes = {}

        coord_dict = self.to_dict
        for dim in "xyz":
            if len(coord_dict[dim]) > 1:
                diff = coord_dict[dim][1:] - coord_dict[dim][0:-1]

                diff_left = np.pad(diff, ((1, 0)), mode="edge")
                diff_right = np.pad(diff, ((0, 1)), mode="edge")

                diff_avg = 0.5 * (diff_left + diff_right)
                cell_sizes[dim] = diff_avg
            else:
                cell_sizes[dim] = 1

        return cell_sizes

    @cached_property
    def cell_size_meshgrid(self):
        """Returns an N-dimensional grid where N is the number of coordinate arrays that have more than one
        element. Each grid element corresponds to the size of the mesh cell in N-dimensions and 1 for N=0."""
        coord_dict = self.to_dict

        cell_size_meshgrid = np.squeeze(np.ones(tuple(len(coord_dict[dim]) for dim in "xyz")))
        meshgrid_elements = [
            size for dim, size in self.cell_sizes.items() if len(coord_dict[dim]) > 1
        ]

        if len(meshgrid_elements) > 1:
            meshgrid = np.meshgrid(*meshgrid_elements, indexing="ij")
            for idx in range(0, len(meshgrid)):
                cell_size_meshgrid *= np.reshape(meshgrid[idx], cell_size_meshgrid.shape)
        elif len(meshgrid_elements) == 1:
            cell_size_meshgrid = meshgrid_elements[0]

        return cell_size_meshgrid

    def _interp_from_xarray(
        self,
        array: Union[SpatialDataArray, ScalarFieldDataArray],
        interp_method: InterpMethod,
        fill_value: Union[Literal["extrapolate"], float] = "extrapolate",
    ) -> Union[SpatialDataArray, ScalarFieldDataArray]:
        """
        Similar to ``xarrray.DataArray.interp`` with 2 enhancements:

            1) Check if the coordinate of the supplied data are in monotonically increasing order.
            If they are, apply the faster ``assume_sorted=True``.

            2) For axes of single entry, instead of error, apply ``isel()`` along the axis.

        Parameters
        ----------
        array : Union[:class:`.SpatialDataArray`, :class:`.ScalarFieldDataArray`]
            Supplied scalar dataset
        interp_method : :class:`.InterpMethod`
            Interpolation method.
        fill_value : Union[Literal['extrapolate'], float] = "extrapolate"
            Value used to fill in for points outside the data range. If set to 'extrapolate',
            values will be extrapolated into those regions using the "nearest" method.

        Returns
        -------
        Union[:class:`.SpatialDataArray`, :class:`.ScalarFieldDataArray`]
            The interpolated spatial dataset.

        Note
        ----
        This method is called from a :class:`Coords` instance with the array to be interpolated as
        an argument, not the other way around.
        """
        # Check which axes need interpolation or selection
        interp_ax = []
        isel_ax = []
        for ax in "xyz":
            if array.sizes[ax] == 1:
                isel_ax.append(ax)
            else:
                interp_ax.append(ax)

        # apply iselection for the axis containing single entry
        if len(isel_ax) > 0:
            array = array.isel({ax: [0] * len(self.to_dict[ax]) for ax in isel_ax})
            array = array.assign_coords({ax: self.to_dict[ax] for ax in isel_ax})
            if len(interp_ax) == 0:
                return array

        # Apply interp for the rest
        is_sorted = all(np.all(np.diff(array.coords[f]) > 0) for f in interp_ax)
        interp_param = {
            "method": interp_method,
            "assume_sorted": is_sorted,
            "kwargs": {
                "bounds_error": False,
                "fill_value": fill_value,
            },
        }

        # Mark extrapolated points with nan's to fill in later
        if fill_value == "extrapolate" and interp_method != "nearest":
            interp_param["kwargs"]["fill_value"] = np.nan

        # interpolation
        interp_array = array.interp({ax: self.to_dict[ax] for ax in interp_ax}, **interp_param)

        # Fill in nan's with nearest values
        if fill_value == "extrapolate" and interp_method != "nearest":
            interp_param["method"] = "nearest"
            interp_param["kwargs"]["fill_value"] = "extrapolate"
            nearest_array = array.interp({ax: self.to_dict[ax] for ax in interp_ax}, **interp_param)
            interp_array.values[:] = np.where(
                np.isnan(interp_array.values), nearest_array.values, interp_array.values
            )

        return interp_array

    def _interp_from_unstructured(
        self,
        array: UnstructuredGridDatasetType,
        interp_method: InterpMethod,
        fill_value: Union[Literal["extrapolate"], float] = "extrapolate",
    ) -> SpatialDataArray:
        """
        Interpolate from untructured grid onto a Cartesian one.

        Parameters
        ----------
        array : Union[class:`.TriangularGridDataset`, class:`.TetrahedralGridDataset`]
            Supplied scalar dataset
        interp_method : :class:`.InterpMethod`
            Interpolation method.
        fill_value : Union[Literal['extrapolate'], float] = "extrapolate"
            Value used to fill in for points outside the data range. If set to 'extrapolate',
            values will be extrapolated into those regions using the "nearest" method.

        Returns
        -------
        :class:`.SpatialDataArray`
            The interpolated spatial dataset.

        Note
        ----
        This method is called from a :class:`Coords` instance with the array to be interpolated as
        an argument, not the other way around.
        """
        interp_array = array.interp(
            **{ax: self.to_dict[ax] for ax in "xyz"}, method=interp_method, fill_value=fill_value
        )

        return interp_array

    def spatial_interp(
        self,
        array: Union[SpatialDataArray, ScalarFieldDataArray, UnstructuredGridDatasetType],
        interp_method: InterpMethod,
        fill_value: Union[Literal["extrapolate"], float] = "extrapolate",
    ) -> Union[SpatialDataArray, ScalarFieldDataArray]:
        """
        Similar to ``xarrray.DataArray.interp`` with 2 enhancements:

            1) (if input data is an ``xarrray.DataArray``) Check if the coordinate of the supplied
            data are in monotonically increasing order. If they are, apply the faster
            ``assume_sorted=True``.

            2) Data is assumed invariant along zero-size dimensions (if any).

        Parameters
        ----------
        array : Union[
                :class:`.SpatialDataArray`,
                :class:`.ScalarFieldDataArray`,
                :class:`.TriangularGridDataset`,
                :class:`.TetrahedralGridDataset`,
        ]
            Supplied scalar dataset
        interp_method : :class:`.InterpMethod`
            Interpolation method.
        fill_value : Union[Literal['extrapolate'], float] = "extrapolate"
            Value used to fill in for points outside the data range. If set to 'extrapolate',
            values will be extrapolated into those regions using the "nearest" method.

        Returns
        -------
        Union[:class:`.SpatialDataArray`, :class:`.ScalarFieldDataArray`]
            The interpolated spatial dataset.

        Note
        ----
        This method is called from a :class:`Coords` instance with the array to be interpolated as
        an argument, not the other way around.
        """

        # Check for empty dimensions
        result_coords = dict(self.to_dict)
        if any(len(v) == 0 for v in result_coords.values()):
            if isinstance(array, (SpatialDataArray, ScalarFieldDataArray)):
                for c in array.coords:
                    if c not in result_coords:
                        result_coords[c] = array.coords[c].values
            result_shape = tuple(len(v) for v in result_coords.values())
            result = DataArray(np.empty(result_shape, dtype=array.dtype), coords=result_coords)
            return result

        # interpolation
        if isinstance(array, UnstructuredGridDataset):
            return self._interp_from_unstructured(
                array=array, interp_method=interp_method, fill_value=fill_value
            )
        else:
            return self._interp_from_xarray(
                array=array, interp_method=interp_method, fill_value=fill_value
            )


class FieldGrid(Tidy3dBaseModel):
    """Holds the grid data for a single field.

    Example
    -------
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-1, 1, 11)
    >>> z = np.linspace(-1, 1, 12)
    >>> coords = Coords(x=x, y=y, z=z)
    >>> field_grid = FieldGrid(x=coords, y=coords, z=coords)
    """

    x: Coords = pd.Field(
        ...,
        title="X Positions",
        description="x,y,z coordinates of the locations of the x-component of a vector field.",
    )

    y: Coords = pd.Field(
        ...,
        title="Y Positions",
        description="x,y,z coordinates of the locations of the y-component of a vector field.",
    )

    z: Coords = pd.Field(
        ...,
        title="Z Positions",
        description="x,y,z coordinates of the locations of the z-component of a vector field.",
    )


class YeeGrid(Tidy3dBaseModel):
    """Holds the yee grid coordinates for each of the E and H positions.

    Example
    -------
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-1, 1, 11)
    >>> z = np.linspace(-1, 1, 12)
    >>> coords = Coords(x=x, y=y, z=z)
    >>> field_grid = FieldGrid(x=coords, y=coords, z=coords)
    >>> yee_grid = YeeGrid(E=field_grid, H=field_grid)
    >>> Ex_coords = yee_grid.E.x
    """

    E: FieldGrid = pd.Field(
        ...,
        title="Electric Field Grid",
        description="Coordinates of the locations of all three components of the electric field.",
    )

    H: FieldGrid = pd.Field(
        ...,
        title="Electric Field Grid",
        description="Coordinates of the locations of all three components of the magnetic field.",
    )

    @property
    def grid_dict(self):
        """The Yee grid coordinates associated to various field components as a dictionary."""
        return {
            "Ex": self.E.x,
            "Ey": self.E.y,
            "Ez": self.E.z,
            "Hx": self.H.x,
            "Hy": self.H.y,
            "Hz": self.H.z,
        }


class Grid(Tidy3dBaseModel):
    """Contains all information about the spatial positions of the FDTD grid.

    Example
    -------
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-1, 1, 11)
    >>> z = np.linspace(-1, 1, 12)
    >>> coords = Coords(x=x, y=y, z=z)
    >>> grid = Grid(boundaries=coords)
    >>> centers = grid.centers
    >>> sizes = grid.sizes
    >>> yee_grid = grid.yee
    """

    boundaries: Coords = pd.Field(
        ...,
        title="Boundary Coordinates",
        description="x,y,z coordinates of the boundaries between cells, defining the FDTD grid.",
    )

    @staticmethod
    def _avg(coords1d: Coords1D):
        """Return average positions of an array of 1D coordinates."""
        return (coords1d[1:] + coords1d[:-1]) / 2.0

    @staticmethod
    def _min(coords1d: Coords1D):
        """Return minus positions of 1D coordinates."""
        return coords1d[:-1]

    @property
    def centers(self) -> Coords:
        """Return centers of the cells in the :class:`Grid`.

        Returns
        -------
        :class:`Coords`
            centers of the FDTD cells in x,y,z stored as :class:`Coords` object.

        Example
        -------
        >>> x = np.linspace(-1, 1, 10)
        >>> y = np.linspace(-1, 1, 11)
        >>> z = np.linspace(-1, 1, 12)
        >>> coords = Coords(x=x, y=y, z=z)
        >>> grid = Grid(boundaries=coords)
        >>> centers = grid.centers
        """
        return Coords(**{key: self._avg(val) for key, val in self.boundaries.to_dict.items()})

    @property
    def sizes(self) -> Coords:
        """Return sizes of the cells in the :class:`Grid`.

        Returns
        -------
        :class:`Coords`
            Sizes of the FDTD cells in x,y,z stored as :class:`Coords` object.

        Example
        -------
        >>> x = np.linspace(-1, 1, 10)
        >>> y = np.linspace(-1, 1, 11)
        >>> z = np.linspace(-1, 1, 12)
        >>> coords = Coords(x=x, y=y, z=z)
        >>> grid = Grid(boundaries=coords)
        >>> sizes = grid.sizes
        """
        return Coords(**{key: np.diff(val) for key, val in self.boundaries.to_dict.items()})

    @property
    def num_cells(self) -> Tuple[int, int, int]:
        """Return sizes of the cells in the :class:`Grid`.

        Returns
        -------
        tuple[int, int, int]
            Number of cells in the grid in the x, y, z direction.

        Example
        -------
        >>> x = np.linspace(-1, 1, 10)
        >>> y = np.linspace(-1, 1, 11)
        >>> z = np.linspace(-1, 1, 12)
        >>> coords = Coords(x=x, y=y, z=z)
        >>> grid = Grid(boundaries=coords)
        >>> Nx, Ny, Nz = grid.num_cells
        """
        return [len(self.boundaries.dict()[dim]) - 1 for dim in "xyz"]

    @property
    def min_size(self) -> float:
        """Return minimal cells size in all dimensions.

        Returns
        -------
        float
            Minimal cells size in all dimensions.
        """
        return float(min(min(sizes) for sizes in self.sizes.to_list))

    @property
    def max_size(self) -> float:
        """Return maximal cells size in all dimensions.

        Returns
        -------
        float
            Maximal cells size in all dimensions.
        """
        return float(max(max(sizes) for sizes in self.sizes.to_list))

    @property
    def info(self) -> Dict:
        """Dictionary collecting various properties of the grids."""
        num_cells = self.num_cells
        total_cells = int(np.prod(num_cells))
        return {
            "Nx": num_cells[0],
            "Ny": num_cells[1],
            "Nz": num_cells[2],
            "grid_points": total_cells,
            "min_grid_size": self.min_size,
            "max_grid_size": self.max_size,
            "computational_complexity": total_cells / self.min_size,
        }

    @property
    def _primal_steps(self) -> Coords:
        """Return primal steps of the cells in the :class:`Grid`.

        Returns
        -------
        :class:`Coords`
            Distances between each of the cell boundaries along each dimension.
        """
        return self.sizes

    @property
    def _dual_steps(self) -> Coords:
        """Return dual steps of the cells in the :class:`Grid`.

        Returns
        -------
        :class:`Coords`
            Distances between each of the cell centers along each dimension, with periodicity
            applied.
        """

        primal_steps = {dim: self._primal_steps.dict()[dim] for dim in "xyz"}
        dsteps = {key: (psteps + np.roll(psteps, 1)) / 2 for (key, psteps) in primal_steps.items()}

        return Coords(**dsteps)

    @property
    def yee(self) -> YeeGrid:
        """Return the :class:`YeeGrid` defining the yee cell locations for this :class:`Grid`.


        Returns
        -------
        :class:`YeeGrid`
            Stores coordinates of all of the components on the yee lattice.

        Example
        -------
        >>> x = np.linspace(-1, 1, 10)
        >>> y = np.linspace(-1, 1, 11)
        >>> z = np.linspace(-1, 1, 12)
        >>> coords = Coords(x=x, y=y, z=z)
        >>> grid = Grid(boundaries=coords)
        >>> yee_cells = grid.yee
        >>> Ex_positions = yee_cells.E.x
        """
        yee_e_kwargs = {key: self._yee_e(axis=axis) for axis, key in enumerate("xyz")}
        yee_h_kwargs = {key: self._yee_h(axis=axis) for axis, key in enumerate("xyz")}

        yee_e = FieldGrid(**yee_e_kwargs)
        yee_h = FieldGrid(**yee_h_kwargs)
        return YeeGrid(E=yee_e, H=yee_h)

    def __getitem__(self, coord_key: str) -> Coords:
        """quickly get the grid element by grid[key]."""

        coord_dict = {
            "centers": self.centers,
            "sizes": self.sizes,
            "boundaries": self.boundaries,
            "Ex": self.yee.E.x,
            "Ey": self.yee.E.y,
            "Ez": self.yee.E.z,
            "Hx": self.yee.H.x,
            "Hy": self.yee.H.y,
            "Hz": self.yee.H.z,
        }
        if coord_key not in coord_dict:
            raise SetupError(f"key {coord_key} not found in grid with {list(coord_dict.keys())} ")

        return coord_dict.get(coord_key)

    def _yee_e(self, axis: Axis):
        """E field yee lattice sites for axis."""

        boundary_coords = self.boundaries.to_dict

        # initially set all to the minus bounds
        yee_coords = {key: self._min(val) for key, val in boundary_coords.items()}

        # average the axis index between the cell boundaries
        key = "xyz"[axis]
        yee_coords[key] = self._avg(boundary_coords[key])

        return Coords(**yee_coords)

    def _yee_h(self, axis: Axis):
        """H field yee lattice sites for axis."""

        boundary_coords = self.boundaries.to_dict

        # initially set all to centers
        yee_coords = {key: self._avg(val) for key, val in boundary_coords.items()}

        # set the axis index to the minus bounds
        key = "xyz"[axis]
        yee_coords[key] = self._min(boundary_coords[key])

        return Coords(**yee_coords)

    def discretize_inds(self, box: Box, extend: bool = False) -> List[Tuple[int, int]]:
        """Start and stopping indexes for the cells that intersect with a :class:`Box`.

        Parameters
        ----------
        box : :class:`Box`
            Rectangular geometry within simulation to discretize.
        extend : bool = False
            If ``True``, ensure that the returned indexes extend sufficiently in every direction to
            be able to interpolate any field component at any point within the ``box``, for field
            components sampled on the Yee grid.

        Returns
        -------
        List[Tuple[int, int]]
            The (start, stop) indexes of the cells that intersect with ``box`` in each of the three
            dimensions.
        """

        pts_min, pts_max = box.bounds
        boundaries = self.boundaries

        inds_list = []

        # for each dimension
        for axis, (pt_min, pt_max) in enumerate(zip(pts_min, pts_max)):
            bound_coords = np.array(boundaries.to_list[axis])
            if pt_min > pt_max:
                raise AssertionError("min point was greater than max point")

            # index of smallest coord greater than pt_max
            inds_gt_pt_max = np.where(bound_coords > pt_max)[0]
            ind_max = len(bound_coords) - 1 if len(inds_gt_pt_max) == 0 else inds_gt_pt_max[0]

            # index of largest coord less than or equal to pt_min
            inds_leq_pt_min = np.where(bound_coords <= pt_min)[0]
            ind_min = 0 if len(inds_leq_pt_min) == 0 else inds_leq_pt_min[-1]

            # handle extensions
            if ind_max > ind_min and extend:
                # Left side
                if box.bounds[0][axis] < self.centers.to_list[axis][ind_min]:
                    # Box bounds on the left side are to the left of the closest grid center
                    ind_min -= 1

                # We always need an extra pixel on the right for the tangential components
                ind_max += 1

            # store indexes
            inds_list.append((ind_min, ind_max))

        return inds_list

    def extended_subspace(
        self,
        axis: Axis,
        ind_beg: int = 0,
        ind_end: int = 0,
        periodic: bool = True,
    ) -> Coords1D:
        """Pick a subspace of 1D boundaries within ``range(ind_beg, ind_end)``. If any indexes lie
        outside of the grid boundaries array, padding is used based on the boundary conditions.
        For periodic BCs, the zeroth and last element of the grid boundaries are identified.
        For other BCs, the zeroth and last element of the boundaries are a reflection plane.

        Parameters
        ----------
        axis : Axis
            Axis index along which to pick the subspace.
        ind_beg : int = 0
            Starting index for the subspace.
        ind_end : int = 0
            Ending index for the subspace.
        periodic : bool = True
            Whether to pad out of bounds indexes with a periodic or reflected pattern.

        Returns
        -------
        Coords1D
            The subspace of the grid along ``axis``.
        """

        coords = self.boundaries.to_list[axis]
        padded_coords = coords
        num_cells = coords.size - 1

        reverse = True
        while ind_beg < 0:
            if periodic or not reverse:
                offset = padded_coords[0] - coords[-1]
                padded_coords = np.concatenate([coords[:-1] + offset, padded_coords])
                reverse = True
            else:
                offset = padded_coords[0] + coords[0]
                padded_coords = np.concatenate([offset - coords[:0:-1], padded_coords])
                reverse = False
            ind_beg += num_cells
            ind_end += num_cells

        reverse = True
        while ind_end >= padded_coords.size:
            if periodic or not reverse:
                offset = padded_coords[-1] - coords[0]
                padded_coords = np.concatenate([padded_coords, coords[1:] + offset])
                reverse = True
            else:
                offset = padded_coords[-1] + coords[-1]
                padded_coords = np.concatenate([padded_coords, offset - coords[-2::-1]])
                reverse = False

        return padded_coords[ind_beg:ind_end]

    def snap_to_box_zero_dim(self, box: Box):
        """Snap a grid to an exact box position for dimensions for which the box is size 0.
        If the box location is outside of the grid, an error is raised.

        Parameters
        ----------
        box : :class:`Box`
            Box to use for the zero dim check.

        Returns
        -------
        class:`Grid`
            Snapped copy of the grid.
        """

        boundary_dict = self.boundaries.to_dict.copy()
        for dim, center, size in zip("xyz", box.center, box.size):
            # Overwrite grid boundaries with box center if box is size 0 along dimension
            if size == 0:
                if boundary_dict[dim][0] > center or boundary_dict[dim][-1] < center:
                    raise ValueError("Cannot snap grid to box center outside of grid domain.")
                boundary_dict[dim] = np.array([center, center])
        return self.updated_copy(boundaries=Coords(**boundary_dict))

    def _translated_copy(self, vector: Coordinate) -> Grid:
        """Translate the grid by a vector. Not officially supported as resulting
        grid may not be aligned with original Yee grid."""
        boundaries = Coords(
            x=self.boundaries.x + vector[0],
            y=self.boundaries.y + vector[1],
            z=self.boundaries.z + vector[2],
        )
        return self.updated_copy(boundaries=boundaries)

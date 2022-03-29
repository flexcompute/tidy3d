"""Defines the FDTD grid."""
from typing import Tuple, List, Union

import numpy as np  # pylint:disable=unused-import
import pydantic as pd

from .base import Tidy3dBaseModel, TYPE_TAG_STR
from .types import Array, Axis
from .geometry import Box
from ..log import SetupError

# data type of one dimensional coordinate array.
Coords1D = Array[float]


class GridSpec(Tidy3dBaseModel):
    """Specification for non-uniform grid along a given dimension.

    Example
    -------
    >>> grid_spec = GridSpec(min_steps_per_wvl=16, max_scale=1.1)
    """

    min_steps_per_wvl: float = pd.Field(
        10.0,
        title="Minimum Steps Per Wavelength",
        description="A nonuniform grid is generated with at least the minimum number of mesh steps "
        "per wavelength in the materia in every structure.",
    )

    max_scale: float = pd.Field(
        1.4,
        title="Maximum Grid Size Scaling",
        description="Sets the maximum ratio between any two consecutive grid steps.",
    )


# Allowed values for ``Simulation.grid_size``
GridSize = Union[pd.PositiveFloat, List[pd.PositiveFloat], GridSpec]


class Bounds1D(Coords1D):
    """1D coords to be used as grid boundaries, with methods to generate automatically from
    simulation parameters."""

    # pylint:disable=too-many-arguments
    @classmethod
    def _make_bounds(cls, grid_size, center, size, symmetry, structures, wvl):
        """Make uniform or nonuniform boundaries depending on grid_size input."""

        if isinstance(grid_size, float):
            bound_coords = cls._from_uniform_dl(grid_size, center, size, symmetry)
        elif isinstance(grid_size, list):
            bound_coords = cls._from_nonuinform_dl(grid_size, center, size)
        elif isinstance(grid_size, GridSpec):
            cent_nu, size_nu = center, size
            if symmetry != 0:
                # If symmetry present, only mesh starting from center
                cent_nu += size / 4
                size_nu /= 2
            bound_coords = cls._from_min_steps(grid_size, cent_nu, size_nu, structures, wvl)

        # Enforce a symmetric grid by reflecting the boundaries around center
        if symmetry != 0:
            bound_coords = bound_coords[bound_coords >= center]
            bound_coords = np.append(2 * center - bound_coords[:0:-1], bound_coords)

        return bound_coords

    @classmethod
    def _from_uniform_dl(cls, dl, center, size, symmetry):
        """Creates coordinate boundaries with uniform mesh (dl is float).
        Center if symmetry present."""

        num_cells = int(np.floor(size / dl))

        # Make sure there's at least one cell
        num_cells = max(num_cells, 1)

        # snap to grid, recenter
        size_snapped = dl * num_cells
        bound_coords = center + np.linspace(-size_snapped / 2, size_snapped / 2, num_cells + 1)

        # Offset to center if symmetry present
        if symmetry != 0:
            bound_coords += center - bound_coords[bound_coords.size // 2]

        return bound_coords

    @classmethod
    def _from_nonuinform_dl(cls, dl, center, size):
        """Creates coordinate boundaries with non-uniform mesh (dl is arraylike).
        These are always centered on the supplied center."""

        # get bounding coordinates
        dl = np.array(dl)
        bound_coords = np.array([np.sum(dl[:i]) for i in range(len(dl) + 1)])

        # place the middle boundary at the center of the simulation along dimension
        bound_coords += center - bound_coords[bound_coords.size // 2]

        # chop off any coords outside of simulation bounds
        bound_min = center - size / 2
        bound_max = center + size / 2
        bound_coords = bound_coords[bound_coords <= bound_max]
        bound_coords = bound_coords[bound_coords >= bound_min]

        # if not extending to simulation bounds, repeat beginning and end
        dl_min = dl[0]
        dl_max = dl[-1]
        while bound_coords[0] - dl_min >= bound_min:
            bound_coords = np.insert(bound_coords, 0, bound_coords[0] - dl_min)
        while bound_coords[-1] + dl_max <= bound_max:
            bound_coords = np.append(bound_coords, bound_coords[-1] + dl_max)

        return bound_coords

    # pylint:disable=too-many-arguments,unused-argument
    @classmethod
    def _from_min_steps(cls, grid_spec, center, size, structures, wvl):
        """Creates coordinate boundaries with non-uniform mesh based on required minimum steps
        per wavelength."""
        return


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
    def to_list(self):
        """Return a list of the three Coord1D objects."""
        return list(self.dict(exclude={TYPE_TAG_STR}).values())


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
        yee_grid_dict = {
            "Ex": self.E.x,
            "Ey": self.E.y,
            "Ez": self.E.z,
            "Hx": self.H.x,
            "Hy": self.H.y,
            "Hz": self.H.z,
        }
        return yee_grid_dict


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
        return Coords(
            **{
                key: self._avg(val)
                for key, val in self.boundaries.dict(exclude={TYPE_TAG_STR}).items()
            }
        )

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
        return Coords(
            **{
                key: np.diff(val)
                for key, val in self.boundaries.dict(exclude={TYPE_TAG_STR}).items()
            }
        )

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
        return [
            coords1d.size - 1 for coords1d in self.boundaries.dict(exclude={TYPE_TAG_STR}).values()
        ]

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

        primal_steps = self._primal_steps.dict(exclude={TYPE_TAG_STR})
        dsteps = {}
        for (key, psteps) in primal_steps.items():
            dsteps[key] = (psteps + np.roll(psteps, 1)) / 2

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

        boundary_coords = self.boundaries.dict(exclude={TYPE_TAG_STR})

        # initially set all to the minus bounds
        yee_coords = {key: self._min(val) for key, val in boundary_coords.items()}

        # average the axis index between the cell boundaries
        key = "xyz"[axis]
        yee_coords[key] = self._avg(boundary_coords[key])

        return Coords(**yee_coords)

    def _yee_h(self, axis: Axis):
        """H field yee lattice sites for axis."""

        boundary_coords = self.boundaries.dict(exclude={TYPE_TAG_STR})

        # initially set all to centers
        yee_coords = {key: self._avg(val) for key, val in boundary_coords.items()}

        # set the axis index to the minus bounds
        key = "xyz"[axis]
        yee_coords[key] = self._min(boundary_coords[key])

        return Coords(**yee_coords)

    def discretize_inds(self, box: Box) -> List[Tuple[int, int]]:
        """Start and stopping indexes for the cells that intersect with a :class:`Box`.

        Parameters
        ----------
        box : :class:`Box`
            Rectangular geometry within simulation to discretize.

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
        for axis_label, pt_min, pt_max in zip("xyz", pts_min, pts_max):
            bound_coords = boundaries.dict()[axis_label]
            assert pt_min <= pt_max, "min point was greater than max point"

            # index of smallest coord greater than than pt_max
            inds_gt_pt_max = np.where(bound_coords > pt_max)[0]
            ind_max = len(bound_coords) - 1 if len(inds_gt_pt_max) == 0 else inds_gt_pt_max[0]

            # index of largest coord less than or equal to pt_min
            inds_leq_pt_min = np.where(bound_coords <= pt_min)[0]
            ind_min = 0 if len(inds_leq_pt_min) == 0 else inds_leq_pt_min[-1]

            # store indexes
            inds_list.append((ind_min, ind_max))

        return inds_list

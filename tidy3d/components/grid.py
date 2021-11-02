""" defines the FDTD grid """
import numpy as np

from .base import Tidy3dBaseModel
from .types import Array, Axis

""" Grid data """

# data type of one dimensional coordinate array.
Coords1D = Array[float]


class Coords(Tidy3dBaseModel):
    """Holds data about a set of x,y,z positions on a grid.

    Parameters
    ----------
    x : np.ndarray
        Positions of coordinates along x direction.
    y : np.ndarray
        Positions of coordinates along y direction.
    z : np.ndarray
        Positions of coordinates along z direction.

    Example
    -------
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-1, 1, 11)
    >>> z = np.linspace(-1, 1, 12)
    >>> coords = Coords(x=x, y=y, z=z)
    """

    x: Coords1D
    y: Coords1D
    z: Coords1D


class FieldGrid(Tidy3dBaseModel):
    """Holds the grid data for a single field.

    Parameters
    ----------
    x : :class:`Coords`
        x,y,z coordinates of the locations of the x-component of the field.
    y : :class:`Coords`
        x,y,z coordinates of the locations of the y-component of the field.
    z : :class:`Coords`
        x,y,z coordinates of the locations of the z-component of the field.

    Example
    -------
    >>> x = np.linspace(-1, 1, 10)
    >>> y = np.linspace(-1, 1, 11)
    >>> z = np.linspace(-1, 1, 12)
    >>> coords = Coords(x=x, y=y, z=z)
    >>> field_grid = FieldGrid(x=coords, y=coords, z=coords)
    """

    x: Coords
    y: Coords
    z: Coords


class YeeGrid(Tidy3dBaseModel):
    """Holds the yee grid coordinates for each of the E and H positions.

    Parameters
    ----------
    E : :class:`FieldGrid`
        x,y,z coordinates of the locations of all three components of the electric field.
    H : :class:`FieldGrid`
        x,y,z coordinates of the locations of all three components of the magnetic field.

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

    E: FieldGrid
    H: FieldGrid


class Grid(Tidy3dBaseModel):
    """Contains all information about the spatial positions of the FDTD grid.

    Parameters
    ----------
    boundaries : :class:`Coords`
        x,y,z coordinates of the boundaries between cells, defining the FDTD grid.

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

    boundaries: Coords

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
        return Coords(**{key: self._avg(val) for key, val in self.boundaries.dict().items()})

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
        return Coords(**{key: np.diff(val) for key, val in self.boundaries.dict().items()})

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
            Distances between each of the cell centers along each dimension.
        """
        return Coords(**{key: np.diff(val) for key, val in self.centers.dict().items()})

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

    def _yee_e(self, axis: Axis):
        """E field yee lattice sites for axis."""

        boundary_coords = self.boundaries.dict()

        # initially set all to the minus bounds
        yee_coords = {key: self._min(val) for key, val in boundary_coords.items()}

        # average the axis index between the cell boundaries
        key = "xyz"[axis]
        yee_coords[key] = self._avg(boundary_coords[key])

        return Coords(**yee_coords)

    def _yee_h(self, axis: Axis):
        """E field yee lattice sites for axis."""

        boundary_coords = self.boundaries.dict()

        # initially set all to the minus bounds
        yee_coords = {key: self._avg(val) for key, val in boundary_coords.items()}

        # average the axis index between the cell boundaries
        key = "xyz"[axis]
        yee_coords[key] = self._min(boundary_coords[key])

        return Coords(**yee_coords)

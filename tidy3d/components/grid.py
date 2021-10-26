""" defines the FDTD grid """
import numpy as np

from .base import Tidy3dBaseModel
from .types import Array, Axis

""" Grid data """

# type of one dimensional coordinate array
Coords1D = Array[float]


class Coords(Tidy3dBaseModel):
    """Holds data about a set of x,y,z positions on a grid"""

    x: Coords1D
    y: Coords1D
    z: Coords1D


class FieldGrid(Tidy3dBaseModel):
    """Holds the grid data for a single field"""

    x: Coords
    y: Coords
    z: Coords


class YeeGrid(Tidy3dBaseModel):
    """Holds the yee grid coordinates for each of the E and H positions"""

    E: FieldGrid
    H: FieldGrid


class Grid(Tidy3dBaseModel):
    """contains all information about the spatial positions of the FDTD grid"""

    boundaries: Coords

    @staticmethod
    def _avg(coords1d: Coords1D):
        """average an array of 1D coordinates"""
        return (coords1d[1:] + coords1d[:-1]) / 2.0

    @staticmethod
    def _min(coords1d: Coords1D):
        """get minus positions of 1D coordinates"""
        return coords1d[:-1]

    @property
    def centers(self):
        """get centers of the coords"""
        return Coords(**{key: self._avg(val) for key, val in self.boundaries.dict().items()})

    @property
    def cell_sizes(self):
        """get centers of the coords"""
        return Coords(**{key: np.diff(val) for key, val in self.boundaries.dict().items()})

    @property
    def yee(self):
        """yee cell grid"""
        yee_e_kwargs = {key: self._yee_e(axis=axis) for axis, key in enumerate("xyz")}
        yee_h_kwargs = {key: self._yee_h(axis=axis) for axis, key in enumerate("xyz")}

        yee_e = FieldGrid(**yee_e_kwargs)
        yee_h = FieldGrid(**yee_h_kwargs)
        return YeeGrid(E=yee_e, H=yee_h)

    def _yee_e(self, axis: Axis):  #
        """E field yee lattice sites for axis"""

        boundary_coords = self.boundaries.dict()

        # initially set all to the minus bounds
        yee_coords = {key: self._min(val) for key, val in boundary_coords.items()}

        # average the axis index between the cell boundaries
        key = "xyz"[axis]
        yee_coords[key] = self._avg(boundary_coords[key])

        return Coords(**yee_coords)

    def _yee_h(self, axis: Axis):
        """E field yee lattice sites for axis"""

        boundary_coords = self.boundaries.dict()

        # initially set all to the minus bounds
        yee_coords = {key: self._avg(val) for key, val in boundary_coords.items()}

        # average the axis index between the cell boundaries
        key = "xyz"[axis]
        yee_coords[key] = self._min(boundary_coords[key])

        return Coords(**yee_coords)

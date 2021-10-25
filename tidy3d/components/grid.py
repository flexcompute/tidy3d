""" defines the FDTD grid """
from abc import ABC, abstractmethod
from typing import Union, List, Tuple

import numpy as np
import pydantic as pd

from .base import Tidy3dBaseModel
from .types import Array, Coordinate, Size, Size1D, Literal, Axis
from ..log import log

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

    cell_boundaries: Coords

    @staticmethod
    def _avg(coords1d: Coords1D):
        """average an array of 1D coordinates"""
        return (coords1d[1:] + coords1d[:-1]) / 2.0

    @staticmethod
    def _min(coords1d: Coords1D):
        """get minus positions of 1D coordinates"""
        return coords1d[:-1]

    @property
    def cell_centers(self):
        """get centers of the coords"""
        return Coords(**{key: self._avg(val) for key, val in self.cell_boundaries.dict().items()})

    @property
    def cell_sizes(self):
        """get centers of the coords"""
        return Coords(**{key: np.diff(val) for key, val in self.cell_boundaries.dict().items()})

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

        boundary_coords = self.cell_boundaries.dict()

        # initially set all to the minus bounds
        yee_coords = {key: self._min(val) for key, val in boundary_coords.items()}

        # average the axis index between the cell boundaries
        key = "xyz"[axis]
        yee_coords[key] = self._avg(boundary_coords[key])

        return Coords(**yee_coords)

    def _yee_h(self, axis: Axis):
        """E field yee lattice sites for axis"""

        boundary_coords = self.cell_boundaries.dict()

        # initially set all to the minus bounds
        yee_coords = {key: self._avg(val) for key, val in boundary_coords.items()}

        # average the axis index between the cell boundaries
        key = "xyz"[axis]
        yee_coords[key] = self._min(boundary_coords[key])

        return Coords(**yee_coords)


# """ 1D grid specifications """


# class GridSpec1D(Tidy3dBaseModel, ABC):
#     """defines the grid along one of the dimensions"""

#     @abstractmethod
#     def get_coords(self, center: float, size: Size1D, wvl_mat_min: float = None) -> Coords1D:
#         """return array of 1D coords for self based on specifications"""

#     @staticmethod
#     def make_coords_dl(center: float, size: Size1D, dl: float) -> Array[float]:
#         """creates evenly spaced coords"""
#         size_snapped = dl * np.floor(size / dl)
#         if size_snapped != size:
#             log.warning(f"grid size ({dl}) is not commensurate with simulation size ({size})")
#         return center + np.linspace(-size_snapped / 2, size_snapped, dl)

""" Grid Specification """


class GridSpec(GridSpec1D):
    """grid defined by the minimum number of points per wavelength inside any :class:`Medium`"""

    pts_per_wvl: pd.PositiveFloat
    nonuniform: Literal[False] = False  # note: when this is supported, make it a bool type = False


GridSpecType = Union[GridSpec, float]

""" Not used """

# class GridSpec1DAuto(GridSpec1D):
#     """1d grid defined by the minimum number of points per wavelength inside any :class:`Medium`"""

#     pts_per_wvl: pd.PositiveFloat
#     nonuniform: Literal[False] = False  # note: when this is supported, make it a true bool

#     def get_coords(self, center: float, size: Size1D, wvl_mat_min: float = None) -> Coords1D:
#         """return array of 1D coords for self based on specifications"""
#         dl_min = wvl_mat_min / self.pts_per_wvl
#         dl_commensurate = size / np.ceil(size / dl_min)
#         return self.make_coords_dl(center=center, size=size, dl=dl_commensurate)


# class GridSpec1DResolution(GridSpec1D):
#     """1d grid defined by number of points per micron"""

#     pts_per_um: pd.PositiveFloat

#     def get_coords(self, center: float, size: Size1D, wvl_mat_min: float = None) -> Coords1D:
#         """return array of 1D coords for self based on specifications"""
#         dl = 1.0 / self.pts_per_um
#         return self.make_coords_dl(center=center, size=size, dl=dl)


# class GridSpec1DSize(GridSpec1D):
#     """1d grid defined by the size of a point"""

#     dl: pd.PositiveFloat

#     def get_coords(self, center: float, size: Size1D, wvl_mat_min: float = None) -> Coords1D:
#         """return array of 1D coords for self based on specifications"""
#         return self.make_coords_dl(center=center, size=size, dl=self.dl)


# class GridSpec1DCoords(GridSpec1D):
#     """1d grid directly defined by the coordinates of the boundaries between adjacent points"""

#     coords: Coords1D

#     def get_coords(self, center: float, size: Size1D, wvl_mat_min: float = None) -> Array[float]:
#         """return array of 1D coords for self based on specifications"""
#         assert (
#             self.coords[0] >= center - size / 2
#         ), "coords dont extend beyond sim bounds in - direction"
#         assert (
#             self.coords[-1] <= center + size / 2
#         ), "coords dont extend beyond sim bounds in + direction"
#         return self.coords


# GridSpec1DType = Union[GridSpec1DAuto, GridSpec1DResolution, GridSpec1DSize, GridSpec1DCoords]
# GridSpec1DType = GridSpec1DSize  # for now only allow this one

""" Global grid specifications """


# class GridSpec(Tidy3dBaseModel):
#     """defines the global grid"""

#     x: GridSpec1DType
#     y: GridSpec1DType
#     z: GridSpec1DType

#     # TODO: We should see if we can safely increase courant to 0.99
#     subpixel: bool = True  # should this really be here? more of a structure thing
#     courant: pd.confloat(gt=0.0, le=1.0) = 0.9

#     def get_coords(self, center: Coordinate, size: Size, wvl_mat_min: float = None) -> Coords:
#         """get coordinates given the global grid specs"""
#         x0, y0, z0 = center
#         Lx, Ly, Lz = size
#         coords_x = self.x.get_coords(center=x0, size=Lx, wvl_mat_min=wvl_mat_min)
#         coords_y = self.y.get_coords(center=y0, size=Ly, wvl_mat_min=wvl_mat_min)
#         coords_z = self.z.get_coords(center=z0, size=Lz, wvl_mat_min=wvl_mat_min)
#         return Coords(x=coords_x, y=coords_y, z=coords_z)

# pylint: disable=invalid-name
""" defines objects in space """

from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import pydantic
import numpy as np
import holoviews as hv

from .base import Tidy3dBaseModel
from .types import Numpy, Bound, Size, Coordinate, Axis, Coordinate2D, Literal, Vertices

BOUND_EPS = 1e-3  # expand bounds by this much
NUM_PTS_RADIUS = 20  # number of edges around circular shapes
PLOT_BUFFER = 1.0  # add this around extents of .visualize()


class Geometry(Tidy3dBaseModel, ABC):
    """abstract base class, defines where something exists in space"""

    @abstractmethod
    def _get_bounds(self) -> Bound:
        """Returns bounding box for this geometry, must implement for subclasses"""

    @abstractmethod
    def _get_crosssection_polygons(self, position: float, axis: Axis) -> List[Vertices]:
        """returns list of polygon vertices that intersect with plane"""

    def _intersects(self, other) -> bool:
        """method determining whether two geometries' bounds intersect"""

        self_bmin, self_bmax = self._get_bounds()
        other_bmin, other_bmax = other._get_bounds()  # pylint: disable=protected-access

        # are all of other's minimum coordinates less than self's maximum coordinate?
        in_minus = all(o <= s for (s, o) in zip(self_bmax, other_bmin))

        # are all of other's maximum coordinates greater than self's minum coordinate?
        in_plus = all(o >= s for (s, o) in zip(self_bmin, other_bmax))

        # for intersection of bounds, both must be true
        return in_minus and in_plus

    def visualize(self, axis: Axis):
        """make interactive plot"""

        hv.extension("bokeh")

        _, (xlabel, ylabel) = self._pop_axis("xyz", axis=axis)

        b_min, b_max = self._get_bounds()
        _, (x_min, y_min) = self._pop_axis(b_min, axis=axis)
        _, (x_max, y_max) = self._pop_axis(b_max, axis=axis)
        extents = (
            x_min - PLOT_BUFFER,
            y_min - PLOT_BUFFER,
            x_max + PLOT_BUFFER,
            y_max + PLOT_BUFFER,
        )

        def poly_fn(position=0):
            vertices_list = self._get_crosssection_polygons(position, axis=axis)
            polygons = []
            for vertices in vertices_list:
                xs = [x for (x, y) in vertices]
                ys = [y for (x, y) in vertices]
                polygons.append({"x": xs, "y": ys})
            poly = hv.Polygons(polygons, extents=extents)
            return poly

        pos_dim = hv.Dimension("position", range=(-3.0, 3.0), step=0.0001)
        dmap = hv.DynamicMap(poly_fn, kdims=[pos_dim])
        return dmap.opts(xlabel=xlabel, ylabel=ylabel)

    @staticmethod
    def _pop_axis(coord: Coordinate, axis: Axis) -> Tuple[float, Coordinate2D]:
        """separate axis coordinate from planar coordinate"""
        plane_vals = list(coord)
        axis_val = plane_vals.pop(axis)
        return axis_val, plane_vals


""" geometry subclasses """


class Box(Geometry):
    """rectangular Box (has size and center)"""

    center: Coordinate = (0.0, 0.0, 0.0)
    size: Size
    type: Literal["Box"] = "Box"

    def _get_bounds(self) -> Bound:
        """sets bounds based on size and center"""
        size = self.size
        center = self.center
        coord_min = tuple(c - s / 2 - BOUND_EPS for (s, c) in zip(size, center))
        coord_max = tuple(c + s / 2 + BOUND_EPS for (s, c) in zip(size, center))
        return (coord_min, coord_max)

    def _get_crosssection_polygons(self, position: float, axis: Axis) -> List[Vertices]:
        """returns list of polygon vertices that intersect with plane"""
        z0, (x0, y0) = self._pop_axis(self.center, axis=axis)
        Lz, (Lx, Ly) = self._pop_axis(self.size, axis=axis)
        if np.abs(z0 - position) > Lz / 2:
            return []
        rect_vertices = [
            (x0 - Lx / 2, y0 - Ly / 2),
            (x0 + Lx / 2, y0 - Ly / 2),
            (x0 + Lx / 2, y0 + Ly / 2),
            (x0 - Lx / 2, y0 + Ly / 2),
        ]
        return [rect_vertices]


class Sphere(Geometry):
    """A sphere geometry (radius and center)"""

    radius: pydantic.NonNegativeFloat
    center: Coordinate = (0.0, 0.0, 0.0)
    type: Literal["Sphere"] = "Sphere"

    def _get_bounds(self):
        coord_min = tuple(c - self.radius for c in self.center)
        coord_max = tuple(c + self.radius for c in self.center)
        return (coord_min, coord_max)

    def _get_crosssection_polygons(self, position: float, axis: Axis) -> List[Vertices]:
        """returns list of polygon vertices that intersect with plane"""
        z0, (x0, y0) = self._pop_axis(self.center, axis=axis)
        dist_center = np.abs(position - z0)
        if dist_center > self.radius:
            return []
        radius_intersect = np.sqrt(self.radius ** 2 - dist_center ** 2)
        phis = np.linspace(0, 2 * np.pi, NUM_PTS_RADIUS)
        xs = x0 + radius_intersect * np.cos(phis)
        ys = y0 + radius_intersect * np.sin(phis)
        circle_vertices = list(zip(xs, ys))
        return [circle_vertices]


class Cylinder(Geometry):
    """A Cylinder geometry (radius, center, height, axis)"""

    center: Coordinate = (0.0, 0.0, 0.0)
    radius: pydantic.NonNegativeFloat
    length: pydantic.NonNegativeFloat
    axis: Axis = 2
    type: Literal["Cylinder"] = "Cylinder"

    def _get_bounds(self):
        coord_min = list(c - self.radius for c in self.center)
        coord_max = list(c + self.radius for c in self.center)
        coord_min[self.axis] = self.center[self.axis] - self.length / 2.0
        coord_max[self.axis] = self.center[self.axis] + self.length / 2.0
        return (tuple(coord_min), tuple(coord_max))

    def _get_crosssection_polygons(self, position: float, axis: Axis) -> List[Vertices]:
        """returns list of polygon vertices that intersect with plane"""
        z0, (x0, y0) = self._pop_axis(self.center, axis=axis)
        dist_center = np.abs(position - z0)
        if axis == self.axis:
            return self._get_crosssection_top(dist_center, x0, y0)
        return self._get_crosssection_side(dist_center, x0, y0, axis)

    def _get_crosssection_top(self, dist_center: float, x0: float, y0: float) -> List[Vertices]:
        """get cross section when plane normal axis is cylinder axis"""
        if dist_center > self.length / 2:
            return []
        phis = np.linspace(0, 2 * np.pi, NUM_PTS_RADIUS)
        xs = x0 + self.radius * np.cos(phis)
        ys = y0 + self.radius * np.sin(phis)
        circle_vertices = [[x, y] for (x, y) in zip(xs, ys)]
        return [circle_vertices]

    def _get_crosssection_side(
        self, dist_center, x0: float, y0: float, axis: axis
    ) -> List[Vertices]:
        """get cross section when plane normal axis is not cylinder axis"""
        if dist_center > self.radius:
            return []
        radius_intersect = np.sqrt(self.radius ** 2 - dist_center ** 2)
        sizes = [2 * radius_intersect, 2 * radius_intersect, 2 * radius_intersect]
        sizes[self.axis] = self.length
        _, (Lx, Ly) = self._pop_axis(sizes, axis=axis)
        rect_vertices = [
            (x0 - Lx / 2, y0 - Ly / 2),
            (x0 + Lx / 2, y0 - Ly / 2),
            (x0 + Lx / 2, y0 + Ly / 2),
            (x0 - Lx / 2, y0 + Ly / 2),
        ]
        return [rect_vertices]


class PolySlab(Geometry):
    """A polygon with vertices and bounds in plane"""

    vertices: Vertices
    slab_bounds: Tuple[float, float]
    axis: Axis = 2
    sidewall_angle_rad: float = 0  # note, not supported yet
    dilation: float = 0  # note, not supported yet
    type: Literal["PolySlab"] = "PolySlab"

    def _get_bounds(self):

        # get the min and max points in polygon plane
        xpoints = tuple(c[0] for c in self.vertices)
        ypoints = tuple(c[1] for c in self.vertices)
        xmin, xmax = min(xpoints), max(xpoints)
        ymin, ymax = min(ypoints), max(ypoints)

        # create min and max coordinates for polygon in 2D
        coord_min = [xmin, ymin]
        coord_max = [xmax, ymax]

        # insert the slab bounds at the specified `axis`
        z_min, z_max = self.slab_bounds
        coord_min.insert(self.axis, z_min)
        coord_max.insert(self.axis, z_max)

        return (tuple(coord_min), tuple(coord_max))

    def _get_crosssection_polygons(self, position: float, axis: Axis) -> List[Vertices]:
        """returns list of polygon vertices that intersect with plane"""
        if axis == self.axis:
            return self._get_crosssection_top(position)
        return self._get_crosssection_side(position, axis)

    def _get_crosssection_top(self, position: float) -> List[Vertices]:
        """get cross section when plane normal axis is slab axis"""

        z_min, z_max = self.slab_bounds
        if (z_min > position) or (z_max < position):
            return []
        return [self.vertices]

    def _get_crosssection_side(self, position: float, axis: Axis) -> List[Vertices]:
        """get cross section when plane normal axis is not slab axis"""

        z_min, z_max = self.slab_bounds
        iverts_b, iverts_f = self._find_intersecting_vertices(position, axis)
        ints_y = self._find_intersecting_ys(iverts_b, iverts_f, position)

        # make polygon with intersections and z axis information
        polys = []
        for i in range(len(ints_y) // 2):

            # consecutive smaller and larger y points, respectively, assumed material between them
            # y1 = np.float64(ints_y[2 * i])
            # y2 = np.float64(ints_y[2 * i + 1])
            y1 = ints_y[2 * i]
            y2 = ints_y[2 * i + 1]
            poly = [(y1, z_min), (y2, z_min), (y2, z_max), (y1, z_max)]

            polys.append(np.array(poly))

        return polys

    def _find_intersecting_vertices(self, position: float, axis: Axis) -> Tuple[Numpy, Numpy]:
        """find pairs of forward and backwards vertices where interescets"""

        vertices_b = np.array(self.vertices)

        # if the first coordinate refers to bounds, need to flip the vertices x,y
        if (axis == 2) or ((self.axis == 2) and (axis == 1)):
            vertices_b = np.roll(vertices_b, shift=1, axis=1)

        # get the forward vertices
        vertices_f = np.roll(vertices_b, shift=1, axis=0)

        # find which segments intersect
        intersects_b = np.logical_and((vertices_f[:, 0] <= position), (vertices_b[:, 0] > position))
        intersects_f = np.logical_and((vertices_b[:, 0] <= position), (vertices_f[:, 0] > position))
        intersects_segment = np.logical_or(intersects_b, intersects_f)
        iverts_b = vertices_b[intersects_segment]
        iverts_f = vertices_f[intersects_segment]

        return iverts_b, iverts_f

    @staticmethod
    def _find_intersecting_ys(iverts_b: Numpy, iverts_f: Numpy, position: float) -> List[float]:
        """for each intersecting segment, find intersection point (in y) assuming straight line"""

        ints_y = []
        for (vertices_f, vertices_b) in zip(iverts_b, iverts_f):
            # find the interescting y
            x1, y1 = vertices_f
            x2, y2 = vertices_b
            slope = (y2 - y1) / (x2 - x1)
            y = y1 + slope * (position - x1)
            ints_y.append(y)

        # sort the intersections just to be safe
        ints_y.sort()
        return ints_y


GeometryFields = (Box, Sphere, Cylinder, PolySlab)
GeometryType = Union[GeometryFields]

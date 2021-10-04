""" defines objects in space """

from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import pydantic
import numpy as np
import holoviews as hv
import matplotlib as mpl

from .base import Tidy3dBaseModel
from .types import Literal, Numpy, Bound, Size, Coordinate, Axis
from .types import Coordinate2D, Vertices, AxesSubplot
from .viz import add_ax_if_none, GeoParams

BOUND_EPS = 1e-3  # expand bounds by this much
NUM_PTS_RADIUS = 101  # number of edges around circular shapes
PLOT_BUFFER = 0.3  # add this around extents of .visualize()


class Geometry(Tidy3dBaseModel, ABC):
    """abstract base class, defines where something exists in space"""

    @abstractmethod
    def get_bounds(self) -> Bound:
        """Returns bounding box for this geometry, must implement for subclasses"""

    def _get_bounding_box(self):
        """Get Box() representing bounding box of geometry"""
        (xmin, ymin, zmin), (xmax, ymax, zmax) = self.get_bounds()
        Lx = xmax - xmin
        Ly = ymax - ymin
        Lz = zmax - zmin
        x0 = (xmax + xmin) / 2.0
        y0 = (ymax + ymin) / 2.0
        z0 = (zmax + zmin) / 2.0
        return Box(center=(x0, y0, z0), size=(Lx, Ly, Lz))

    @abstractmethod
    def _get_crosssection_polygons(self, position: float, axis: Axis) -> List[Vertices]:
        """returns list of polygon vertices that intersect with plane"""

    @abstractmethod
    def is_inside(self, x, y, z) -> bool:
        """returns True if (x,y,z) is inside of geometry"""

    def intersects(self, other) -> bool:
        """method determining whether two geometries' bounds intersect"""

        self_bmin, self_bmax = self.get_bounds()
        other_bmin, other_bmax = other.get_bounds()  # pylint: disable=protected-access

        # are all of other's minimum coordinates less than self's maximum coordinate?
        in_minus = all(o <= s for (s, o) in zip(self_bmax, other_bmin))

        # are all of other's maximum coordinates greater than self's minum coordinate?
        in_plus = all(o >= s for (s, o) in zip(self_bmin, other_bmax))

        # for intersection of bounds, both must be true
        return in_minus and in_plus

    def intersects_plane(self, position: float, axis: Axis) -> bool:
        """whether self intersects plane at `position` along normal `axis`"""
        (zmin, zmax), _ = self._pop_bounds(axis=axis)
        is_above_bottom = position >= zmin
        is_below_top = position <= zmax
        return is_above_bottom and is_below_top

    @staticmethod
    def _pop_axis(coord: Coordinate, axis: Axis) -> Tuple[float, Coordinate2D]:
        """separate axis coordinate from planar coordinate"""
        plane_vals = list(coord)
        axis_val = plane_vals.pop(axis)
        return axis_val, plane_vals

    def _pop_bounds(
        self, axis: Axis
    ) -> Tuple[Tuple[float, float], Tuple[Tuple[float, float], Tuple[float, float]]]:
        """returns min and max bounds in plane normal to `axis`"""
        b_min, b_max = self.get_bounds()
        zmin, (xmin, ymin) = self._pop_axis(b_min, axis=axis)
        zmax, (xmax, ymax) = self._pop_axis(b_max, axis=axis)
        return (zmin, zmax), ((xmin, ymin), (xmax, ymax))

    def _get_plot_labels(self, axis: Axis) -> Tuple[str, str]:
        """get x, y axis labels for cross section plots"""
        _, (xlabel, ylabel) = self._pop_axis("xyz", axis=axis)
        return xlabel, ylabel

    def _get_plot_extents(
        self, axis: Axis, buffer: float = PLOT_BUFFER
    ) -> Tuple[float, float, float, float]:
        """get xmin, ymin, xmax, ymax extents for cross section plots"""
        _, ((xmin, ymin), (xmax, ymax)) = self._pop_bounds(axis=axis)
        extents = (
            xmin - buffer,
            ymin - buffer,
            xmax + buffer,
            ymax + buffer,
        )
        return extents

    def _add_ax_labels_lims(
        self, axis: Axis, ax: AxesSubplot, buffer: float = PLOT_BUFFER
    ) -> AxesSubplot:
        """sets the x,y labels based on axis and the extends based on self.bounds"""
        xlabel, ylabel = self._get_plot_labels(axis=axis)
        (xmin, ymin, xmax, ymax) = self._get_plot_extents(axis=axis, buffer=buffer)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    @add_ax_if_none
    def plot(  # pylint: disable=too-many-arguments
        self,
        position: float,
        axis: Axis,
        ax: AxesSubplot = None,
        **plot_params: dict,
    ) -> AxesSubplot:
        """plot the geometry on the plane"""
        plot_params_new = GeoParams().update_params(**plot_params)

        vertices_list = self._get_crosssection_polygons(position, axis=axis)
        for vertices in vertices_list:
            patch = mpl.patches.Polygon(vertices, **plot_params_new)
            ax.add_patch(patch)
        ax = self._add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_aspect("equal")
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")
        return ax

    def visualize(self, axis: Axis):
        """make interactive plot"""

        hv.extension("bokeh")

        xlabel, ylabel = self._get_plot_labels(axis=axis)
        extents = self._get_plot_extents(axis=axis)

        def poly_fn(position=0):
            """returns hv.polygons as function of sliding bar position"""
            vertices_list = self._get_crosssection_polygons(position, axis=axis)
            polygons = []
            for vertices in vertices_list:
                xs = [x for (x, _) in vertices]
                ys = [y for (_, y) in vertices]
                polygons.append({"x": xs, "y": ys})
            poly = hv.Polygons(polygons, extents=extents)
            return poly

        pos_dim = hv.Dimension("position", range=(-3.0, 3.0), step=0.0001)
        dmap = hv.DynamicMap(poly_fn, kdims=[pos_dim])
        return dmap.opts(xlabel=xlabel, ylabel=ylabel)


""" geometry subclasses """


class Box(Geometry):
    """rectangular Box (has size and center)"""

    center: Coordinate = (0.0, 0.0, 0.0)
    size: Size
    type: Literal["Box"] = "Box"

    def get_bounds(self) -> Bound:
        """sets bounds based on size and center"""
        size = self.size
        center = self.center
        coord_min = tuple(c - s / 2 - BOUND_EPS for (s, c) in zip(size, center))
        coord_max = tuple(c + s / 2 + BOUND_EPS for (s, c) in zip(size, center))
        return (coord_min, coord_max)

    def is_inside(self, x, y, z) -> bool:
        """returns True if (x,y,z) is inside of geometry"""

        x0, y0, z0 = self.center
        Lx, Ly, Lz = self.size

        dist_x = np.abs(x - x0)
        dist_y = np.abs(y - y0)
        dist_z = np.abs(z - z0)

        return (dist_x < Lx / 2) * (dist_y < Ly / 2) * (dist_z < Lz / 2)

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

    @property
    def geometry(self):
        """return a `Box` representation of self
        useful for subclasses of Box, eg. FieldMonitor.geometry -> Box"""
        return Box(center=self.center, size=self.size)


class Sphere(Geometry):
    """A sphere geometry (radius and center)"""

    radius: pydantic.NonNegativeFloat
    center: Coordinate = (0.0, 0.0, 0.0)
    type: Literal["Sphere"] = "Sphere"

    def get_bounds(self):
        """returns bounding box (rmin, rmax)"""
        coord_min = tuple(c - self.radius for c in self.center)
        coord_max = tuple(c + self.radius for c in self.center)
        return (coord_min, coord_max)

    def is_inside(self, x, y, z) -> bool:
        """returns True if (x,y,z) is inside of geometry"""
        x0, y0, z0 = self.center
        dist_x = np.abs(x - x0)
        dist_y = np.abs(y - y0)
        dist_z = np.abs(z - z0)
        return dist_x ** 2 + dist_y ** 2 + dist_z ** 2 <= self.radius ** 2

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

    def get_bounds(self):
        """returns bounding box (rmin, rmax)"""
        coord_min = list(c - self.radius for c in self.center)
        coord_max = list(c + self.radius for c in self.center)
        coord_min[self.axis] = self.center[self.axis] - self.length / 2.0
        coord_max[self.axis] = self.center[self.axis] + self.length / 2.0
        return (tuple(coord_min), tuple(coord_max))

    def is_inside(self, x, y, z) -> bool:
        """returns True if (x,y,z) is inside of geometry"""
        z0, (x0, y0) = self._pop_axis(self.center, axis=self.axis)
        z, (x, y) = self._pop_axis((x, y, z), axis=self.axis)
        dist_x = np.abs(x - x0)
        dist_y = np.abs(y - y0)
        dist_z = np.abs(z - z0)
        return (dist_z <= self.length) * (dist_x ** 2 + dist_y ** 2 <= self.radius ** 2)

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

    def get_bounds(self):
        """returns bounding box (rmin, rmax)"""

        # get the min and max points in polygon plane
        xpoints = tuple(c[0] for c in self.vertices)
        ypoints = tuple(c[1] for c in self.vertices)
        xmin, xmax = min(xpoints), max(xpoints)
        ymin, ymax = min(ypoints), max(ypoints)

        # create min and max coordinates for polygon in 2D
        coord_min = [xmin, ymin]
        coord_max = [xmax, ymax]

        # insert the slab bounds at the specified `axis`
        zmin, zmax = self.slab_bounds
        coord_min.insert(self.axis, zmin)
        coord_max.insert(self.axis, zmax)

        return (tuple(coord_min), tuple(coord_max))

    def is_inside(self, x, y, z) -> bool:
        """returns True if (x,y,z) is inside of geometry"""
        z, (x, y) = self._pop_axis((x, y, z), axis=self.axis)
        zmin, zmax = self.slab_bounds
        path = mpl.path.Path(self.vertices)
        xy_points = np.stack((x, y), axis=1)
        in_polygon_xy = path.contains_points(xy_points)
        return (z >= zmin) * (z <= zmax) * in_polygon_xy

    def _get_crosssection_polygons(self, position: float, axis: Axis) -> List[Vertices]:
        """returns list of polygon vertices that intersect with plane"""
        if axis == self.axis:
            return self._get_crosssection_top(position)
        return self._get_crosssection_side(position, axis)

    def _get_crosssection_top(self, position: float) -> List[Vertices]:
        """get cross section when plane normal axis is slab axis"""

        zmin, zmax = self.slab_bounds
        if (zmin > position) or (zmax < position):
            return []
        return [self.vertices]

    def _get_crosssection_side(self, position: float, axis: Axis) -> List[Vertices]:
        """get cross section when plane normal axis is not slab axis"""

        zmin, zmax = self.slab_bounds
        iverts_b, iverts_f = self._find_intersecting_vertices(position, axis)
        ints_y = self._find_intersecting_ys(iverts_b, iverts_f, position)

        # make polygon with intersections and z axis information
        polys = []
        for y_index in range(len(ints_y) // 2):
            y1 = ints_y[2 * y_index]
            y2 = ints_y[2 * y_index + 1]
            poly = [(y1, zmin), (y2, zmin), (y2, zmax), (y1, zmax)]

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
            x1, y1 = vertices_f
            x2, y2 = vertices_b
            slope = (y2 - y1) / (x2 - x1)
            y = y1 + slope * (position - x1)
            ints_y.append(y)
        ints_y.sort()
        return ints_y


GeometryFields = (Box, Sphere, Cylinder, PolySlab)
GeometryType = Union[GeometryFields]

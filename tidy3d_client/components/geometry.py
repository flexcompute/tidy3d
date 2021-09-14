import pydantic
from abc import ABC, abstractmethod

from .base import Tidy3dBaseModel
from .types import Bound, Size, Coordinate, Axis, Coordinate2D, Any, List, Tuple
from .validators import check_bounds

BOUND_EPS = 1e-3  # expand bounds by this much

""" defines objects in space """

class Geometry(ABC, Tidy3dBaseModel):
    """ abstract base class, defines where something exists in space"""

    @abstractmethod
    def _get_bounds(self) -> Bound:
        """ Returns bounding box for this geometry, must implement for subclasses """
        pass

    def _intersects(self, other) -> bool:
        """ method determining whether two geometries' bounds intersect """

        self_bmin, self_bmax = self._get_bounds()
        other_bmin, other_bmax = other._get_bounds()

        # are all of other's minimum coordinates less than self's maximum coordinate?
        in_minus = all(o <= s for (s, o) in zip(self_bmax, other_bmin))

        # are all of other's maximum coordinates greater than self's minum coordinate?
        in_plus = all(o >= s for (s, o) in zip(self_bmin, other_bmax))

        # for intersection of bounds, both must be true
        return in_minus and in_plus


class GeometryObject(ABC, Tidy3dBaseModel):
    """ signifies an object containing a geometry, (such as structure, source, monitor) """

    geometry: Geometry = None

""" geometry subclasses """

class Box(Geometry):
    """rectangular Box (has size and center)"""

    size: Size
    center: Coordinate = (0.0, 0.0, 0.0)

    def _get_bounds(self) -> Bound:
        """sets bounds based on size and center"""
        size = self.size
        center = self.center
        coord_min = tuple(c - s/2 - BOUND_EPS for (s, c) in zip(size, center))
        coord_max = tuple(c + s/2 + BOUND_EPS for (s, c) in zip(size, center))
        return (coord_min, coord_max)


class Sphere(Geometry):
    radius: pydantic.NonNegativeFloat
    center: Coordinate = (0.0, 0.0, 0.0)

    def _get_bounds(self):
        coord_min = tuple(c - self.radius for c in self.center)
        coord_max = tuple(c + self.radius for c in self.center)
        return (coord_min, coord_max)


class Cylinder(Geometry):
    radius: pydantic.NonNegativeFloat
    length: pydantic.NonNegativeFloat
    center: Coordinate = (0.0, 0.0, 0.0)
    axis: Axis = 2

    def _get_bounds(self):
        coord_min = list(c - self.radius for c in self.center)
        coord_max = list(c + self.radius for c in self.center)
        coord_min[self.axis] = self.center[self.axis] - self.length/2.
        coord_max[self.axis] = self.center[self.axis] + self.length/2.
        return (tuple(coord_min), tuple(coord_max))

class PolySlab(Geometry):
    vertices: List[Coordinate2D]
    slab_bounds: Tuple[float, float]
    axis: Axis = 2
    sidewall_angle_rad: float = 0  # note, not supported yet
    dilation: float = 0            # note, not supported yet

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
        zmin, zmax = self.slab_bounds
        coord_min.insert(self.axis, zmin)
        coord_max.insert(self.axis, zmax)

        return (tuple(coord_min), tuple(coord_max))


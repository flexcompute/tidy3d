import pydantic

from .base import Tidy3dBaseModel
from .types import Bound, Size, Coordinate, Axis, Coordinate2D, Any, List, Tuple
from .validators import check_bounds

BOUND_EPS = 1e-3  # expand bounds by this much

class Geometry(Tidy3dBaseModel):
    """defines where something exists in space"""

    bounds: Bound = None

    def __init__(self, **data: Any):
        """checks the bounds after any Geometry instance is initialized"""
        super().__init__(**data)
        self.bounds = self._get_bounds()
        _bound_validator = check_bounds()

    def _get_bounds(self) -> Bound:
        """ returns bounding box for this geometry """
        raise NotImplementedError(f"Must implement self._get_bounds() for '{type(self).__name__}' geometry")


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

class GeometryObject(Tidy3dBaseModel):
    """ an object with a geometry """

    geometry: Geometry
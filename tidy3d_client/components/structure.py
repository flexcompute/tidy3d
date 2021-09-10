from .base import Tidy3dBaseModel
from .geometry import GeometryObject
from .medium import Medium

class Structure(GeometryObject):
    """An object that interacts with the electromagnetic fields"""

    medium: Medium

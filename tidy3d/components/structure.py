from .base import Tidy3dBaseModel
from .geometry import Geometry
from .medium import Medium

class Structure(Tidy3dBaseModel):
    """An object that interacts with the electromagnetic fields"""

    geometry: Geometry
    medium: Medium
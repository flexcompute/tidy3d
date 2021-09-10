import pydantic

from .base import Tidy3dBaseModel
from .types import Tuple
from .validators import ensure_greater_or_equal
from .geometry import GeometryObject, Box

class SourceTime(Tidy3dBaseModel):
    """Base class describing the time dependence of a source"""

    amplitude: pydantic.NonNegativeFloat = 1.0
    phase: float = 0.0


class Pulse(SourceTime):
    """A general pulse time dependence"""

    freq0: pydantic.PositiveFloat
    fwidth: pydantic.PositiveFloat
    offset: pydantic.NonNegativeFloat = 5.0

    _validate_offset = ensure_greater_or_equal("offset", 2.5)


class Source(GeometryObject):
    """Defines electric and magnetic currents that produce electromagnetic field"""

    geometry: Box
    source_time: SourceTime
    polarization: Tuple[float, float, float]
from .base import Tidy3dBaseModel
from .validators import ensure_greater_or_equal

class Medium(Tidy3dBaseModel):
    """Defines properties of a medium where electromagnetic waves propagate"""

    permittivity: float = 1.0
    conductivity: float = 0.0

    _permittivity_validator = ensure_greater_or_equal("permittivity", 1.0)
    _conductivity_validator = ensure_greater_or_equal("conductivity", 0.0)


import pydantic

from .base import Tidy3dBaseModel
from .types import Literal

class PMLLayer(Tidy3dBaseModel):
    """single layer of a PML (profile and num layers)"""

    profile: Literal["standard", "stable", "absorber"] = "standard"
    num_layers: pydantic.NonNegativeInt = 0
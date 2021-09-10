import pydantic

from .base import Tidy3dBaseModel
from .types import List, Literal
from .geometry import GeometryObject, Box

STORE_VALUES = Literal["E", "H", "flux"]

class Monitor(GeometryObject):
    geometry: Box
    store_values: List[STORE_VALUES] = ["E", "H", "flux"]
    freqs: List[pydantic.NonNegativeFloat] = []
    times: List[pydantic.NonNegativeFloat] = []

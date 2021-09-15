import pydantic
import numpy as np

from .base import Tidy3dBaseModel

class Mode(Tidy3dBaseModel):
	mode_index: pydantic.NonNegativeInt

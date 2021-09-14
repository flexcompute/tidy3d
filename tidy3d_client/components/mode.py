import pydantic
import numpy as np

from .base import Tidy3dBaseModel

class Mode(Tidy3dBaseModel):

	mode_index: pydantic.NonNegativeInt

	def get_profile(xs, ys):
		""" need to implement using mode solver """

		xx, yy = np.meshgrid(xs, ys)
		return np.ones_like(xx, dtype=complex)

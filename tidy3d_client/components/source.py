import pydantic
import numpy as np
from abc import ABC, abstractmethod

from .base import Tidy3dBaseModel
from .types import Tuple, Literal
from .validators import ensure_greater_or_equal, assert_plane
from .geometry import GeometryObject, Box

# def dft(amp_time, time, freq):
#     freq = np.array(freq)[None, :]
#     time = np.array(time)
#     amp_time = np.array(amp_time)
#     phases = 2j * np.pi * freq * time
#     spectrum = np.sum(amp_time * np.exp(phases), axis=0)
#     return dt / np.sqrt(2 * np.pi) * spectrum

class SourceTime(ABC, Tidy3dBaseModel):
    """Base class describing the time dependence of a source"""

    amplitude: pydantic.NonNegativeFloat = 1.0
    phase: float = 0.0

    @abstractmethod
    def amp_time(self, time):
        """ complex amplitude as a function of time """
        pass

class GaussianPulse(SourceTime):
    """A gaussian pulse time dependence"""

    freq0: pydantic.PositiveFloat
    fwidth: pydantic.PositiveFloat
    offset: pydantic.NonNegativeFloat = 5.0

    _validate_offset = ensure_greater_or_equal("offset", 2.5)

    def amp_time(self, time):
        twidth = 1.0 / (2 * np.pi * self.fwidth)
        omega0 = 2 * np.pi * self.freq0

        time_shifted = time - self.offset * twidth

        const = (1j + time_shifted / twidth**2 / omega0)
        offset = np.exp(1j * self.phase)
        oscillation = np.exp(-1j * omega0 * time)
        amp = np.exp(-time_shifted**2 / 2 / twidth**2)

        return const * offset * oscillation * amp


class Source(GeometryObject):
    """Defines electric and magnetic currents that produce electromagnetic field"""

    geometry: Box
    source_time: SourceTime
    polarization: Tuple[float, float, float]

class PlaneWave(Source):
    """Defines Plane Wave Source """

    direction: Literal["+", "-"] 

    _pw_validator = assert_plane(field_name="geometry")


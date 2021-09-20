import pydantic
import numpy as np
from abc import ABC, abstractmethod

from .base import Tidy3dBaseModel
from .types import Tuple, Direction, Polarization, Union
from .validators import assert_plane
from .geometry import Box
from .mode import Mode

# def dft(amp_time, time, freq):
#     freq = np.array(freq)[None, :]
#     time = np.array(time)
#     amp_time = np.array(amp_time)
#     phases = 2j * np.pi * freq * time
#     spectrum = np.sum(amp_time * np.exp(phases), axis=0)
#     return dt / np.sqrt(2 * np.pi) * spectrum

""" Source Times define the time dependence of the source """

class SourceTime(ABC, Tidy3dBaseModel):
    """Base class describing the time dependence of a source"""

    amplitude: pydantic.NonNegativeFloat = 1.0
    phase: float = 0.0

    # @abstractmethod
    def amp_time(self, time):
        """ complex amplitude as a function of time """
        pass

class Pulse(SourceTime, ABC):
    """ Source ramps up and oscillates with freq0 """

    freq0: pydantic.PositiveFloat
    fwidth: pydantic.PositiveFloat
    offset: pydantic.confloat(ge=2.5) = 5.0

class GaussianPulse(Pulse):
    """A gaussian pulse time dependence"""

    def amp_time(self, time):
        twidth = 1.0 / (2 * np.pi * self.fwidth)
        omega0 = 2 * np.pi * self.freq0
        time_shifted = time - self.offset * twidth

        const = (1j + time_shifted / twidth**2 / omega0)
        offset = np.exp(1j * self.phase)
        oscillation = np.exp(-1j * omega0 * time)
        amp = np.exp(-time_shifted**2 / 2 / twidth**2)

        return const * offset * oscillation * amp

class CW(Pulse):
    """ ramping up and holding steady """

    def amp_time(self, time):
        twidth = 1.0 / (2 * np.pi * self.fwidth)
        omega0 = 2 * np.pi * self.freq0
        time_shifted = time - self.offset * twidth

        const = 1.0
        offset = np.exp(1j * self.phase)
        oscillation = np.exp(-1j * omega0 * time)
        amp = 1 / (1 + np.exp(-time_shifted/twidth))

        return const * offset * oscillation * amp

SourceTimeType = Union[GaussianPulse, CW]

""" Source objects """

class Source(Box, ABC):
    """ Template for all sources, all have Box geometry """

    source_time: SourceTimeType

class VolumeSource(Source):
    """ Volume Source with time dependence and polarization """

    polarization: Polarization

class ModeSource(Source):
    """ Modal profile on finite extent plane """

    direction: Direction
    mode: Mode
    _plane_validator = assert_plane()

class DirectionalSource(Source, ABC):
    """ A Planar Source with uni-directional propagation """

    direction: Direction
    polarization: Polarization
    _plane_validator = assert_plane()

    @pydantic.root_validator(allow_reuse=True)
    def polarization_is_orthogonal(cls, values):
        """ ensure we dont allow a polarization parallel to the propagation direction """
        size = values.get('size')
        polarization = values.get('polarization')
        assert size is not None
        assert polarization is not None

        normal_axis_index = size.index(0.0)
        normal_axis = 'xyz'[normal_axis_index]
        polarization_axis = polarization[-1]
        assert normal_axis != polarization_axis, f"Directional source '{cls.__name__}' can not have polarization component ({polarization_axis}) parallel to plane's normal direction ({normal_axis})"
        return values

class PlaneWave(DirectionalSource):
    """ uniform distribution on infinite extent plane """
    pass

class GaussianBeam(DirectionalSource):
    """ guassian distribution on finite extent plane """

    waist_size: Tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat]


# allowable sources to use in Simulation.sources
SourceType = Union[VolumeSource, ModeSource, GaussianBeam]

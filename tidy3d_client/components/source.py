import pydantic
import numpy as np
from abc import ABC, abstractmethod

from .base import Tidy3dBaseModel
from .types import Tuple, List, Direction, Polarization
from .validators import ensure_greater_or_equal, assert_plane
from .geometry import GeometryObject, Box
from .constants import inf
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

    @abstractmethod
    def amp_time(self, time):
        """ complex amplitude as a function of time """
        pass

class Pulse(SourceTime, ABC):
    """ Source ramps up and ramps down """

    freq0: pydantic.PositiveFloat
    fwidth: pydantic.PositiveFloat
    offset: pydantic.NonNegativeFloat = 5.0
    _validate_offset = ensure_greater_or_equal("offset", 2.5)

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


""" Source objects """

class AbstractSource(Box, ABC):
    """ Template for all sources, all have Box geometry """

    source_time: SourceTime

class Source(AbstractSource):
    """ Volume Source with time dependence and polarization """

    polarization: Polarization

class DataSource(AbstractSource):

    data: np.ndarray

    @pydantic.validator("data")
    def is_right_shape(cls, val, values):
        data_shape = val.shape
        dims_data = len(data_shape)
        source_size = values.get("size")
        assert source_size is not None
        dims_box = 3 - source_size.count(0.0)
        assert dims_data == dims_box, f"data must have one axis per non-zero shape of Source, data.shape={data_shape}, source.size={source_size}"
        return val

class ModeSource(AbstractSource):
    """ Modal profile on finite extent plane """

    direction: Direction
    mode: Mode
    _plane_validator = assert_plane()

class DirectionalSource(AbstractSource, ABC):
    """ A Planar Source with uni-directional propagation """

    direction: Direction
    polarization: Polarization
    _plane_validator = assert_plane()

    @pydantic.root_validator()
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

    sigma_plane = Tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat]


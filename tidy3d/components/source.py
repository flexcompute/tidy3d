""" Defines current sources """

from abc import ABC, abstractmethod
from typing import Tuple, Union, Literal

import pydantic
import numpy as np

from .base import Tidy3dBaseModel
from .types import Direction, Polarization, Axis, AxesSubplot, ArrayLike
from .validators import assert_plane
from .geometry import Box
from .mode import Mode
from .viz import add_ax_if_none, SourceParams


class SourceTime(ABC, Tidy3dBaseModel):
    """Base class describing the time dependence of a source"""

    amplitude: pydantic.NonNegativeFloat = 1.0
    phase: float = 0.0

    @abstractmethod
    def amp_time(self, time):
        """complex amplitude as a function of time"""

    @add_ax_if_none
    def plot(  # pylint: disable=rguments-differ
        self, times: ArrayLike, ax: AxesSubplot = None
    ) -> AxesSubplot:
        """plot the time series"""
        times = np.array(times)
        amp_complex = self.amp_time(times)

        times_ps = times / 1e-12
        ax.plot(times_ps, amp_complex.real, color="blueviolet", label="real")
        ax.plot(times_ps, amp_complex.imag, color="crimson", label="imag")
        ax.plot(times_ps, np.abs(amp_complex), color="k", label="abs")
        ax.set_xlabel("time (ps)")
        ax.set_title("source amplitude")
        ax.legend()
        ax.set_aspect("auto")
        return ax


class Pulse(SourceTime, ABC):
    """Source ramps up and oscillates with freq0"""

    freq0: pydantic.PositiveFloat
    fwidth: pydantic.PositiveFloat  # currently standard deviation
    offset: pydantic.confloat(ge=2.5) = 5.0


class GaussianPulse(Pulse):
    """A gaussian pulse time dependence"""

    def amp_time(self, time):
        """complex amplitude as a function of time"""
        twidth = 1.0 / (2 * np.pi * self.fwidth)
        omega0 = 2 * np.pi * self.freq0
        time_shifted = time - self.offset * twidth

        const = 1j + time_shifted / twidth ** 2 / omega0
        offset = np.exp(1j * self.phase)
        oscillation = np.exp(-1j * omega0 * time)
        amp = np.exp(-(time_shifted ** 2) / 2 / twidth ** 2)

        return const * offset * oscillation * amp


class CW(Pulse):
    """ramping up and holding steady"""

    def amp_time(self, time):
        """complex amplitude as a function of time"""
        twidth = 1.0 / (2 * np.pi * self.fwidth)
        omega0 = 2 * np.pi * self.freq0
        time_shifted = time - self.offset * twidth

        const = 1.0
        offset = np.exp(1j * self.phase)
        oscillation = np.exp(-1j * omega0 * time)
        amp = 1 / (1 + np.exp(-time_shifted / twidth))

        return const * offset * oscillation * amp


SourceTimeType = Union[GaussianPulse, CW]

""" Source objects """


class Source(Box, ABC):
    """Template for all sources, all have Box geometry"""

    source_time: SourceTimeType

    @add_ax_if_none
    def plot(  # pylint: disable=invalid-name, arguments-differ
        self, position: float, axis: Axis, ax: AxesSubplot = None, **plot_params: dict
    ) -> AxesSubplot:
        """plot source geometry"""
        plot_params = SourceParams().update_params(**plot_params)
        ax = self.geometry.plot(position=position, axis=axis, ax=ax, **plot_params)
        return ax


class VolumeSource(Source):
    """Volume Source with time dependence and polarization"""

    polarization: Polarization
    type: Literal["VolumeSource"] = "VolumeSource"


class ModeSource(Source):
    """Modal profile on finite extent plane"""

    direction: Direction
    mode: Mode
    type: Literal["ModeSource"] = "ModeSource"
    _plane_validator = assert_plane()


class DirectionalSource(Source, ABC):
    """A Planar Source with uni-directional propagation"""

    direction: Direction
    polarization: Polarization

    _plane_validator = assert_plane()

    @pydantic.root_validator(allow_reuse=True)
    def polarization_is_orthogonal(cls, values):  # pylint: disable=no-self-argument
        """ensure we dont allow a polarization parallel to the propagation direction"""
        size = values.get("size")
        polarization = values.get("polarization")
        assert size is not None
        assert polarization is not None

        normal_axis_index = size.index(0.0)
        normal_axis = "xyz"[normal_axis_index]
        polarization_axis = polarization[-1]
        assert (
            normal_axis != polarization_axis
        ), f"Directional source '{cls.__name__}' "  # pylint: disable=no-member
        " can not have polarization component ({polarization_axis})"
        "parallel to plane's normal direction ({normal_axis})"
        return values


class PlaneWave(DirectionalSource):
    """uniform distribution on infinite extent plane"""

    type: Literal["PlaneWave"] = "PlaneWave"


class GaussianBeam(DirectionalSource):
    """guassian distribution on finite extent plane"""

    waist_size: Tuple[pydantic.NonNegativeFloat, pydantic.NonNegativeFloat]
    type: Literal["GaussianBeam"] = "GaussianBeam"


# allowable sources to use in Simulation.sources
SourceType = Union[VolumeSource, PlaneWave, ModeSource, GaussianBeam]

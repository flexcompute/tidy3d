""" Objects that define how data is recorded from simulation """
from abc import ABC
from typing import List, Union

import numpy as np

from .types import Literal, Axis, AxesSubplot
from .geometry import Box
from .validators import assert_plane
from .mode import Mode
from .viz import add_ax_if_none, MonitorParams


""" Samplers """

FreqSampler = List[float]
TimeSampler = List[int]


def _uniform_arange(start: int, stop: int, step: int) -> TimeSampler:
    """uniform spacing from start to stop with spacing of step"""
    assert start <= stop, "start must not be greater than stop"
    return list(np.arange(start, stop, step))


def _uniform_linspace(start: float, stop: float, num: int) -> FreqSampler:
    """uniform spacing from start to stop with num elements"""
    assert start <= stop, "start must not be greater than stop"
    return list(np.linspace(start, stop, num))


def uniform_time_sampler(t_start: int, t_stop: int, t_step: int = 1) -> TimeSampler:
    """create TimeSampler at evenly spaced time steps"""
    assert isinstance(t_start, int), "`t_start` must be integer for time sampler"
    assert isinstance(t_stop, int), "`t_stop` must be integer for time sampler"
    assert isinstance(t_step, int), "`t_step` must be integer for time sampler"
    times = _uniform_arange(t_start, t_stop, t_step)
    return times


def uniform_freq_sampler(f_start, f_stop, num_freqs) -> FreqSampler:
    """create FreqSampler at evenly spaced frequency points"""
    freqs = _uniform_linspace(f_start, f_stop, num_freqs)
    return freqs


""" Monitors """


class Monitor(Box, ABC):
    """base class for monitors, which all have Box shape"""

    @add_ax_if_none
    def plot(  # pylint: disable=invalid-name, arguments-differ
        self, position: float, axis: Axis, ax: AxesSubplot = None, **plot_params: dict
    ) -> AxesSubplot:
        """plot monitor geometry"""
        plot_params = MonitorParams().update_params(**plot_params)
        ax = self.geometry.plot(position=position, axis=axis, ax=ax, **plot_params)
        return ax


class FreqMonitor(Monitor, ABC):
    """stores data in frequency domain"""

    freqs: FreqSampler


class TimeMonitor(Monitor, ABC):
    """stores data in time domain"""

    times: TimeSampler


class AbstractFieldMonitor(Monitor, ABC):
    """stores data as a function of x,y,z"""


class AbstractFluxMonitor(Monitor, ABC):
    """stores flux data through a surface"""

    _plane_validator = assert_plane()


""" usable """


class FieldMonitor(FreqMonitor, AbstractFieldMonitor):
    """stores EM fields as a function of frequency"""

    type: Literal["FieldMonitor"] = "FieldMonitor"


class FieldTimeMonitor(TimeMonitor, AbstractFieldMonitor):
    """stores EM fields as a function of time"""

    type: Literal["FieldTimeMonitor"] = "FieldTimeMonitor"


class PermittivityMonitor(FreqMonitor, AbstractFieldMonitor):
    """stores permittivity data as a function of frequency"""

    type: Literal["PermittivityMonitor"] = "PermittivityMonitor"


class FluxMonitor(FreqMonitor, AbstractFluxMonitor):
    """Stores flux through a plane as a function of frequency"""

    type: Literal["ModeMonitor"] = "ModeMonitor"


class FluxTimeMonitor(TimeMonitor, AbstractFluxMonitor):
    """Stores flux through a plane as a function of time"""

    type: Literal["FluxTimeMonitor"] = "FluxTimeMonitor"


class ModeMonitor(FreqMonitor):
    """stores modal amplitudes associated with modes"""

    modes: List[Mode]
    type: Literal["ModeMonitor"] = "ModeMonitor"

    _plane_validator = assert_plane()


MonitorFields = (
    FieldMonitor,
    FieldTimeMonitor,
    PermittivityMonitor,
    FluxMonitor,
    FluxTimeMonitor,
    ModeMonitor,
)
MonitorType = Union[MonitorFields]
# Monitor = register_subclasses(MonitorFields)(Monitor)

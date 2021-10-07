""" Objects that define how data is recorded from simulation """
from abc import ABC
from typing import List, Union

import numpy as np

from .types import Literal, Ax, Direction, EMField, Component
from .geometry import Box
from .validators import assert_plane
from .mode import Mode
from .viz import add_ax_if_none, MonitorParams


""" Samplers """

FreqSampler = List[float]
TimeSampler = List[int]


def _uniform_arange(start: int, stop: int, step: int) -> TimeSampler:
    """uniform spacing from start to stop with spacing of step."""
    assert start <= stop, "start must not be greater than stop"
    return list(np.arange(start, stop, step))


def _uniform_linspace(start: float, stop: float, num: int) -> FreqSampler:
    """uniform spacing from start to stop with num elements."""
    assert start <= stop, "start must not be greater than stop"
    return list(np.linspace(start, stop, num))


def uniform_time_sampler(t_start: int, t_stop: int, t_step: int = 1) -> TimeSampler:
    """create times at evenly spaced steps."""
    assert isinstance(t_start, int), "`t_start` must be integer for time sampler"
    assert isinstance(t_stop, int), "`t_stop` must be integer for time sampler"
    assert isinstance(t_step, int), "`t_step` must be integer for time sampler"
    times = _uniform_arange(t_start, t_stop, t_step)
    return times


def uniform_freq_sampler(f_start, f_stop, num_freqs) -> FreqSampler:
    """create frequencies at evenly spaced points."""
    freqs = _uniform_linspace(f_start, f_stop, num_freqs)
    return freqs


""" Monitors """


class Monitor(Box, ABC):
    """base class for monitors, which all have Box shape"""

    @add_ax_if_none
    def plot(self, ax: Ax = None, **kwargs) -> Ax:
        """plot monitor geometry"""
        kwargs = MonitorParams().update_params(**kwargs)
        ax = self.geometry.plot(ax=ax, **kwargs)
        return ax


class FreqMonitor(Monitor, ABC):
    """stores data in frequency domain"""

    freqs: FreqSampler


class TimeMonitor(Monitor, ABC):
    """stores data in time domain"""

    times: TimeSampler


class AbstractFieldMonitor(Monitor, ABC):
    """stores data as a function of x,y,z"""

    component: List[Component] = ["x", "y", "z"]


class AbstractFluxMonitor(Monitor, ABC):
    """stores flux data through a surface"""

    _plane_validator = assert_plane()


""" usable """


class FieldMonitor(FreqMonitor, AbstractFieldMonitor):
    """stores EM fields in volume as a function of frequency.

    Parameters
    ----------
    center: Tuple[float, float, float], optional.
        Center of monitor ``Box``, defaults to (0, 0, 0)
    size: Tuple[float, float, float], optional.
        Size of monitor ``Box``, must have one element = 0.0 to define plane.
    field: List[str], optional
        Electromagnetic field(s) to measure (E, H), defaults to ``['E', 'H']``
    component: List[str], optional
        Directional component to measure in x,y,z, defaults to ``['x','y','z']``.
    freqs: List[float]
        Frequencies to measure fields at at.
    """

    field: List[EMField] = ["E", "H"]
    type: Literal["FieldMonitor"] = "FieldMonitor"


class FieldTimeMonitor(TimeMonitor, AbstractFieldMonitor):
    """stores EM fields as a function of time

    Parameters
    ----------
    center: Tuple[float, float, float], optional.
        Center of monitor ``Box``, defaults to (0, 0, 0)
    size: Tuple[float, float, float], optional.
        Size of monitor ``Box``, must have one element = 0.0 to define plane.
    field: List[str], optional
        Electromagnetic field(s) to measure (E, H), defaults to ``['E', 'H']``
    component: List[str], optional
        Directional component to measure in x,y,z, defaults to ``['x','y','z']``.
    times: List[int]
        Time steps to measure the fields at.
    """

    field: List[EMField] = ["E", "H"]
    type: Literal["FieldTimeMonitor"] = "FieldTimeMonitor"


class PermittivityMonitor(FreqMonitor, AbstractFieldMonitor):
    """stores permittivity data as a function of frequency

    Parameters
    ----------
    center: Tuple[float, float, float], optional.
        Center of monitor ``Box``, defaults to (0, 0, 0)
    size: Tuple[float, float, float], optional.
        Size of monitor ``Box``, must have one element = 0.0 to define plane.
    component: List[str], optional
        Directional component to measure in x,y,z, defaults to ``['x','y','z']``.
    freqs: List[float], optional.
        Frequencies to measure permittivity at. If None, measure at infinite freq.
    """

    freqs: FreqSampler = None
    type: Literal["PermittivityMonitor"] = "PermittivityMonitor"


class FluxMonitor(FreqMonitor, AbstractFluxMonitor):
    """Stores power flux through a plane as a function of frequency.

    Parameters
    ----------
    center: Tuple[float, float, float], optional.
        Center of monitor ``Box``, defaults to (0, 0, 0)
    size: Tuple[float, float, float], optional.
        Size of monitor ``Box``, must have one element = 0.0 to define plane.
    freqs: List[float]
        Frequencies to measure flux at.
    """

    type: Literal["FluxMonitor"] = "FluxMonitor"


class FluxTimeMonitor(TimeMonitor, AbstractFluxMonitor):
    """Stores power flux through a plane as a function of frequency.

    Parameters
    ----------
    center: Tuple[float, float, float], optional.
        Center of monitor ``Box``, defaults to (0, 0, 0)
    size: Tuple[float, float, float], optional.
        Size of monitor ``Box``, must have one element = 0.0 to define plane.
    times: List[int]
        Time steps to measure flux at.
    """

    type: Literal["FluxTimeMonitor"] = "FluxTimeMonitor"


class ModeMonitor(FreqMonitor):
    """stores overlap amplitudes associated with modes.

    Parameters
    ----------
    center: Tuple[float, float, float], optional.
        Center of monitor ``Box``, defaults to (0, 0, 0)
    size: Tuple[float, float, float], optional.
        Size of monitor ``Box``, must have one element = 0.0 to define plane.
    freqs: List[float]
        Frequencies to measure flux at.
    modes: List[``Mode``]
        List of ``Mode`` objects specifying the modal profiles to measure amplitude overalap with.
    """

    direction: List[Direction] = ["+", "-"]
    modes: List[Mode]
    type: Literal["ModeMonitor"] = "ModeMonitor"

    _plane_validator = assert_plane()


# maps monitor type name to monitor type
monitor_type_map = {
    "FieldMonitor": FieldMonitor,
    "FieldTimeMonitor": FieldTimeMonitor,
    "PermittivityMonitor": PermittivityMonitor,
    "FluxMonitor": FluxMonitor,
    "FluxTimeMonitor": FluxTimeMonitor,
    "ModeMonitor": ModeMonitor,
}

MonitorType = Union[tuple(monitor_type_map.values())]

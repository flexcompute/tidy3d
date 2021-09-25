""" Objects that define how data is recorded from simulation """
from abc import ABC
import json

import pydantic
import numpy as np

from .base import Tidy3dBaseModel
from .types import List, Union
from .geometry import Box
from .validators import assert_plane
from .mode import Mode

""" Convenience functions for creating uniformly spaced samplers """


def _uniform_arange(start, stop, step):
    """uniform spacing from start to stop with spacing of step"""
    assert start <= stop, "start must not be greater than stop"
    return list(np.arange(start, stop, step))


def _uniform_linspace(start, stop, num):
    """uniform spacing from start to stop with num elements"""
    assert start <= stop, "start must not be greater than stop"
    return list(np.linspace(start, stop, num))


def uniform_time_sampler(t_start, t_stop, t_step=1):
    """create TimeSampler at evenly spaced time steps"""
    assert isinstance(t_start, int), "`t_start` must be integer for time sampler"
    assert isinstance(t_stop, int), "`t_stop` must be integer for time sampler"
    assert isinstance(t_step, int), "`t_step` must be integer for time sampler"

    times = _uniform_arange(t_start, t_stop, t_step)
    return TimeSampler(times=times)


def uniform_freq_sampler(f_start, f_stop, num_freqs):
    """create FreqSampler at evenly spaced frequency points"""
    freqs = _uniform_linspace(f_start, f_stop, num_freqs)
    return FreqSampler(freqs=freqs)


""" Samplers """


class Sampler(Tidy3dBaseModel, ABC):
    """specifies how the data is sampled as the simulation is run"""


class TimeSampler(Sampler):
    """specifies at what time steps the data is measured"""

    times: List[pydantic.NonNegativeInt]
    _label: str = "times"

    def __len__(self):
        return len(self.times)


class FreqSampler(Sampler):
    """specifies at what frequencies the data is measured using running DFT"""

    freqs: List[pydantic.NonNegativeFloat]
    _label: str = "freqs"

    def __len__(self):
        return len(self.freqs)


# samplers allowed to be used in monitors
SamplerType = Union[TimeSampler, FreqSampler]

""" Monitors """


class Monitor(Box, ABC):
    """base class for monitors, which all have Box shape"""

    mon_type: str

    def __init__(self, **kwargs):
        mon_type = type(self).__name__
        kwargs["mon_type"] = mon_type
        super().__init__(**kwargs)

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, dict):
            mon_type = v.get("mon_type")
            json_string = json.dumps(v)
        else:
            mon_type = v.mon_type
            json_string = v.json()
        cls_type = MonitorMap[mon_type]
        return cls_type.parse_raw(json_string)


class Monitor(Box, ABC):
    """base class for monitors, which all have Box shape"""


class FieldMonitor(Monitor):
    """stores E, H data on the monitor"""

    sampler: SamplerType


class PermittivityMonitor(Monitor):
    """stores permittivity data on the monitor"""

    sampler: FreqSampler


class FluxMonitor(Monitor):
    """Stores flux on a surface"""

    sampler: SamplerType
    _plane_validator = assert_plane()


class ModeMonitor(Monitor):
    """stores amplitudes associated with modes"""

    sampler: FreqSampler
    modes: List[Mode]
    _plane_validator = assert_plane()


# cls_name -> monitors allowed to be used in simulation.monitors
MonitorMap = {
    "FieldMonitor": FieldMonitor,
    "FluxMonitor": FluxMonitor,
    "PermittivityMonitor": PermittivityMonitor,
    "ModeMonitor": ModeMonitor,
}

MonitorType = Union[tuple(MonitorMap.values())]
# from .base import register_subclasses
# Monitor=register_subclasses(MonitorMap)(Monitor())

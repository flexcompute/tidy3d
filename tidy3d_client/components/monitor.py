import pydantic
import numpy as np

from abc import ABC

from .base import Tidy3dBaseModel
from .types import List, Union
from .geometry import GeometryObject, Box
from .validators import assert_plane
from .mode import Mode

""" Domains """

class Domain(Tidy3dBaseModel, ABC):
    """ specifies how the data is stored, whether in time or frequency domain """
    pass

class TimeDomain(Domain):
    """ specifies how often data is sampled in time domain """

    t_start: pydantic.NonNegativeFloat = 0
    t_stop: pydantic.NonNegativeFloat = None
    t_step: pydantic.PositiveInt = 1

    @pydantic.root_validator(allow_reuse=True)
    def stop_gt_start(cls, values):
        """ make sure stop time is greater than start time """
        t_stop = values.get('t_stop')
        t_start = values.get('t_start')
        assert t_start is not None
        if t_stop is not None:
            assert t_stop > t_start, f'`t_stop` (given {t_stop}) must be greater than `t_start` (given {t_start})'
        return values

class FreqDomain(Domain):
    """ specifies at what frequencies data is stored """
    freqs: List[pydantic.NonNegativeFloat]


""" Monitors """

class Monitor(Box, ABC):
    """ base class for monitors, which all have Box shape """
    pass

class FieldMonitor(Monitor):
    """ stores E, H data on the monitor """
    domain: Domain

class FluxMonitor(Monitor):
    """ Stores flux on a surface """
    domain: Domain
    _plane_validator = assert_plane()

class ModeMonitor(Monitor):
    """ stores amplitudes associated with modes """
    domain: FreqDomain
    modes: List[Mode]
    _plane_validator = assert_plane()

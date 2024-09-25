from typing import Any

import autograd.numpy as anp
import pydantic.v1 as pd

from tidy3d.components.monitor import ModeMonitor
from tidy3d.components.types import Direction, FreqArray
from tidy3d.exceptions import ValidationError

from .types import NumberType
from .variables import Variable


class Metric(Variable):
    """
    Base class for all metrics.

    To subclass Metric, you must implement an evaluate() method that takes a SimulationData
    object and returns a scalar value.
    """

    def __repr__(self) -> str:
        return f'{self.type}("{self.monitor_name}")'


class ModeAmp(Metric):
    """
    Metric for calculating the mode coefficient from a ModeMonitor.

    Parameters
    ----------
    monitor_name : str
        The name of the mode monitor.
    freqs : FreqArray
        The frequency array.
    direction : Direction = "+"
        The direction of the mode.
    mode_index : pd.NonNegativeInt = 0
        The index of the mode.

    Examples
    --------
    >>> monitor = ModeMonitor(name="monitor1", freqs=[1.0])
    >>> mode_coeff = ModeAmp.from_mode_monitor(monitor)
    >>> data = SimulationData()  # Assume this is a valid SimulationData object
    >>> result = mode_coeff.evaluate(data)
    """

    monitor_name: str
    freqs: FreqArray
    direction: Direction = "+"
    mode_index: pd.NonNegativeInt = 0

    @pd.validator("freqs", always=True)
    def _single_frequency(cls, val: FreqArray) -> FreqArray:
        if len(val) != 1:
            raise ValidationError("Only a single frequency is supported at the moment.")
        return val

    @classmethod
    def from_mode_monitor(cls, monitor: ModeMonitor):
        return cls(monitor_name=monitor.name, freqs=monitor.freqs, mode_index=0)

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        data = super().evaluate(*args, **kwargs)
        amps = (
            data[self.monitor_name]
            .amps.sel(direction=self.direction, mode_index=self.mode_index)
            .isel(f=0)
        )
        return anp.squeeze(amps.values.tolist())


class ModePower(ModeAmp):
    """
    Metric for calculating the mode power from a ModeMonitor.

    Examples
    --------
    >>> monitor = ModeMonitor(name="monitor1", freqs=[1.0])
    >>> mode_power = ModePower.from_mode_monitor(monitor)
    >>> data = SimulationData()  # Assume this is a valid SimulationData object
    >>> result = mode_power.evaluate(data)
    """

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        amps = super().evaluate(*args, **kwargs)
        return abs(amps) ** 2

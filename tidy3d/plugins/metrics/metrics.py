import abc

import autograd.numpy as anp
import pydantic.v1 as pd

from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.monitor import ModeMonitor
from tidy3d.components.types import Direction, FreqArray
from tidy3d.exceptions import ValidationError

from .base import Expression
from .types import NumberType


class Metric(Expression):
    """
    Base class for all metrics.

    To subclass Metric, you must implement an evaluate() method that takes a SimulationData
    object and returns a scalar value.
    """

    @abc.abstractmethod
    def evaluate(self, data: SimulationData) -> NumberType:
        pass

    def __repr__(self) -> str:
        return f'{self.type}("{self.monitor_name}")'


class ModeCoefficient(Metric):
    """
    Metric for calculating the mode coefficient from a ModeMonitor.

    Attributes
    ----------
    monitor_name : str
        The name of the mode monitor.
    freqs : FreqArray
        The frequency array.
    direction : Direction, default "+"
        The direction of the mode.
    mode_index : pd.NonNegativeInt, default 0
        The index of the mode.

    Methods
    -------
    from_mode_monitor(monitor: ModeMonitor)
        Creates a ModeCoefficient instance from a ModeMonitor.
    evaluate(data: SimulationData) -> NumberType
        Evaluates the mode coefficient from the simulation data.

    Examples
    --------
    >>> monitor = ModeMonitor(name="monitor1", freqs=[1.0])
    >>> mode_coeff = ModeCoefficient.from_mode_monitor(monitor)
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

    def evaluate(self, data: SimulationData) -> NumberType:
        amps = data[self.monitor_name].amps.sel(
            direction=self.direction, mode_index=self.mode_index, f=self.freqs[0]
        )
        return anp.squeeze(amps.values.tolist())


class ModePower(ModeCoefficient):
    """
    Metric for calculating the mode power from a ModeMonitor.

    Methods
    -------
    evaluate(data: SimulationData) -> NumberType
        Evaluates the mode power from the simulation data.

    Examples
    --------
    >>> monitor = ModeMonitor(name="monitor1", freqs=[1.0])
    >>> mode_power = ModePower.from_mode_monitor(monitor)
    >>> data = SimulationData()  # Assume this is a valid SimulationData object
    >>> result = mode_power.evaluate(data)
    """

    def evaluate(self, data: SimulationData) -> NumberType:
        amps = super().evaluate(data)
        return abs(amps) ** 2

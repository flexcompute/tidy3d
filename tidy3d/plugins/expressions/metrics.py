from typing import Any, Optional, Union

import autograd.numpy as np
import pydantic.v1 as pd

from tidy3d.components.monitor import ModeMonitor
from tidy3d.components.types import Direction, FreqArray

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

    Examples
    --------
    >>> monitor = ModeMonitor(name="monitor1", f=[1.0])
    >>> mode_coeff = ModeAmp.from_mode_monitor(monitor)
    >>> data = SimulationData()  # Assume this is a valid SimulationData object
    >>> result = mode_coeff.evaluate(data)
    """

    monitor_name: str = pd.Field(
        ...,
        title="Monitor Name",
        description="The name of the mode monitor. This needs to match the name of the monitor in the simulation.",
    )
    f: Optional[Union[float, FreqArray]] = pd.Field(  # type: ignore
        None,
        title="Frequency Array",
        description="The frequency array. If None, all frequencies in the monitor will be used.",
    )
    direction: Direction = pd.Field(
        "+",
        title="Direction",
        description="The direction of propagation of the mode.",
    )
    mode_index: pd.NonNegativeInt = pd.Field(
        0,
        title="Mode Index",
        description="The index of the mode.",
    )

    @classmethod
    def from_mode_monitor(
        cls, monitor: ModeMonitor, mode_index: int = 0, direction: Direction = "+"
    ):
        return cls(
            monitor_name=monitor.name, f=monitor.freqs, mode_index=mode_index, direction=direction
        )

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        data = super().evaluate(*args, **kwargs)
        amps = data[self.monitor_name].amps.sel(
            direction=self.direction, mode_index=self.mode_index
        )
        if self.f is not None:
            amps = amps.sel(f=list(self.f), method="nearest")
        return np.squeeze(amps.values.tolist())


class ModePower(ModeAmp):
    """
    Metric for calculating the mode power from a ModeMonitor.

    Examples
    --------
    >>> monitor = ModeMonitor(name="monitor1", f=[1.0])
    >>> mode_power = ModePower.from_mode_monitor(monitor)
    >>> data = SimulationData()  # Assume this is a valid SimulationData object
    >>> result = mode_power.evaluate(data)
    """

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        amps = super().evaluate(*args, **kwargs)
        return np.abs(amps) ** 2

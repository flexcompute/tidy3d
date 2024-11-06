from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import autograd.numpy as np
import pydantic.v1 as pd
import xarray as xr

from tidy3d.components.monitor import ModeMonitor
from tidy3d.components.types import Direction, FreqArray

from .base import Expression
from .types import NumberType
from .variables import Variable


def generate_validation_data(expr: Expression) -> dict[str, xr.Dataset]:
    """Generate combined dummy simulation data for all metrics in the expression.

    Parameters
    ----------
    expr : Expression
        The expression containing metrics.

    Returns
    -------
    dict[str, xr.Dataset]
        The combined validation data.
    """
    metrics = set(expr.filter(target_type=Metric))
    combined_data = {k: v for metric in metrics for k, v in metric._validation_data.items()}
    return combined_data


class Metric(Variable, ABC):
    """
    Base class for all metrics.

    To subclass Metric, you must implement an evaluate() method that takes a SimulationData
    object and returns a scalar value.
    """

    @property
    @abstractmethod
    def _validation_data(self) -> Any:
        """Return dummy data for this metric."""

    def __repr__(self) -> str:
        return f'{self.type}("{self.monitor_name}")'


class ModeAmp(Metric):
    """
    Metric for calculating the mode coefficient from a ModeMonitor.

    Examples
    --------
    >>> import tidy3d as td
    >>> monitor = td.ModeMonitor(size=(1, 1, 0), freqs=[2e14], mode_spec=td.ModeSpec(), name="monitor1")
    >>> mode_coeff = ModeAmp.from_mode_monitor(monitor)
    >>> expr = abs(mode_coeff) ** 2
    >>> print(expr)
    (abs(ModeAmp("monitor1")) ** 2)
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
        alias="freqs",
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

    @property
    def _validation_data(self) -> Any:
        """Return dummy data for this metric (complex array of mode amplitudes)."""
        f = np.atleast_1d(self.f).tolist() if self.f is not None else [1.0]
        amps_data = np.random.rand(len(f)) + 1j * np.random.rand(len(f))
        amps = xr.DataArray(
            amps_data.reshape(1, 1, -1),
            coords={
                "direction": [self.direction],
                "mode_index": [self.mode_index],
                "f": f,
            },
        )
        return {self.monitor_name: xr.Dataset({"amps": amps})}

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        data = super().evaluate(*args, **kwargs)
        amps = data[self.monitor_name].amps.sel(
            direction=self.direction, mode_index=self.mode_index
        )
        if self.f is not None:
            f = list(self.f) if isinstance(self.f, tuple) else self.f
            amps = amps.sel(f=f, method="nearest")
        return np.squeeze(amps.data)


class ModePower(ModeAmp):
    """
    Metric for calculating the mode power from a ModeMonitor.

    Examples
    --------
    >>> import tidy3d as td
    >>> monitor = td.ModeMonitor(size=(1, 1, 0), freqs=[2e14], mode_spec=td.ModeSpec(), name="monitor1")
    >>> mode_power = ModePower.from_mode_monitor(monitor)
    """

    def evaluate(self, *args: Any, **kwargs: Any) -> NumberType:
        amps = super().evaluate(*args, **kwargs)
        return np.abs(amps) ** 2

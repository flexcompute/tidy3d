# defines postprocessing classes for `InverseDesign` objects.

import abc
import typing

import autograd.numpy as anp
import pydantic.v1 as pd

import tidy3d as td

from .base import InvdesBaseModel


class PostprocessOperation(InvdesBaseModel, abc.ABC):
    """Abstract base class defining components that make up postprocessing classes."""

    @abc.abstractmethod
    def evaluate(self, sim_data: td.SimulationData) -> float:
        """How to evaluate this operation on a ``SimulationData`` object."""


class GetPowerMode(PostprocessOperation):
    """Grab the power from a ``ModeMonitor`` and apply an optional weight."""

    monitor_name: str = pd.Field(
        ...,
        title="Monitor Name",
        description="Name of the ``ModeMonitor`` corresponding to the ``ModeData`` "
        "that we want to compute power for.",
    )

    direction: td.components.types.Direction = pd.Field(
        None,
        title="Direction",
        description="If specified, selects a specific direction ``'-'`` or ``'+'`` from the "
        "``ModeData`` to include in power. Otherwise, sums over all directions.",
    )

    mode_index: pd.NonNegativeInt = pd.Field(
        None,
        title="Mode Index",
        description="If specified, selects a specific mode index from the ``ModeData`` "
        "to include in power. Otherwise, sums over all mode indices.",
    )

    f: float = pd.Field(
        None,
        title="Frequency",
        description="If specified, selects a specific frequency from the ``ModeData`` "
        "to include in power. Otherwise, sums over all frequencies.",
    )

    weight: float = pd.Field(
        1.0,
        title="Weight",
        description="Weight specifying the contribution of this power to the objective function. ",
    )

    @property
    def sel_kwargs(self) -> dict[str, typing.Any]:
        """Selection kwargs corresponding to the fields."""
        sel_kwargs_all = dict(direction=self.direction, mode_index=self.mode_index, f=self.f)
        return {key: sel for key, sel in sel_kwargs_all.items() if sel is not None}

    def evaluate(self, sim_data: td.SimulationData) -> float:
        """Evaluate this instance when passed a simulation dataset."""

        mnt_data = sim_data[self.monitor_name]

        if not isinstance(mnt_data, td.ModeData):
            raise ValueError(
                "'GetPowerMode' only works with 'ModeData' corresponding to 'ModeMonitor'. "
                f"Monitor name of '{self.monitor_name}' returned data of type {type(mnt_data)}."
            )

        amps = mnt_data.amps
        amp = amps.sel(**self.sel_kwargs)
        powers = abs(amp.values) ** 2
        power = anp.sum(powers)
        return self.weight * power


class WeightedSum(PostprocessOperation):
    """Weighted sum of ``GetPower`` objects."""

    powers: tuple[GetPowerMode, ...] = pd.Field(
        (),
        title="Powers",
        description="Set of objects specifying how to compute and weigh the power from ``ModeData`` outputs.",
    )

    def evaluate(self, sim_data: td.SimulationData) -> float:
        value = 0.0
        for get_power in self.powers:
            value = value + get_power.evaluate(sim_data)

        return value

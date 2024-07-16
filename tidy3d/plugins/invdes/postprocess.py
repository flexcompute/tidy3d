# defines postprocessing classes for `InverseDesign` objects.
from __future__ import annotations

import abc
import typing

import autograd.numpy as anp
import pydantic.v1 as pd

import tidy3d as td

from .base import InvdesBaseModel

PostProcessFnType = typing.Callable[[td.SimulationData], float]


class AbstractPostProcessOperation(InvdesBaseModel, abc.ABC):
    """Abstract base class defining components that make up postprocessing classes."""

    @abc.abstractmethod
    def evaluate(self, sim_data: td.SimulationData) -> float:
        """How to evaluate this operation on a ``SimulationData`` object."""


class CustomPostProcessOperation(AbstractPostProcessOperation):
    """A postprocessing operation to subclass and implement ones own ``evalute`` method for."""

    def evaluate(self, sim_data: td.SimulationData) -> float:
        """How to evaluate this operation on a ``SimulationData`` object."""
        raise NotImplementedError("Must define the 'self.evaluate(sim_data) -> float' method.")

    @classmethod
    def from_function(cls, fn: PostProcessFnType, **kwargs) -> CustomPostProcessOperation:
        """Create a ``CustomPostProcessOperation`` from a function of ``SimulationData``."""
        cls.evaluate = lambda self, sim_data: fn(sim_data)
        obj = cls(**kwargs)
        return obj


class ElementaryPostProcessOperation(AbstractPostProcessOperation, abc.ABC):
    """A postprocess operation that can work on its own."""


class ModePower(ElementaryPostProcessOperation):
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

        if self.monitor_name not in sim_data:
            raise KeyError(f"No data found for monitor with name '{self.monitor_name}'.")

        mnt_data = sim_data[self.monitor_name]

        if not isinstance(mnt_data, td.ModeData):
            raise ValueError(
                "'ModePower' only works with 'ModeData' corresponding to 'ModeMonitor'. "
                f"Monitor name of '{self.monitor_name}' returned data of type {type(mnt_data)}."
            )

        amps = mnt_data.amps
        amp = amps.sel(**self.sel_kwargs)
        powers = abs(amp.values) ** 2
        power = anp.sum(powers)
        return self.weight * power


ElementaryPostProcessOperationType = typing.Union[ModePower]


class CombinedPostProcessOperation(AbstractPostProcessOperation, abc.ABC):
    """A postprocess operation that combines elementary operations."""

    operations: tuple[ElementaryPostProcessOperationType, ...] = pd.Field(
        (),
        title="PostProcessing Operations",
        description="Set of objects specifying the operations combined in this operation.",
    )


class Sum(CombinedPostProcessOperation):
    """Sum of the evaluate outputs of elementary operation objects."""

    def evaluate(self, sim_data: td.SimulationData) -> float:
        """sum the evaluation of each operation."""

        return sum(op.evaluate(sim_data) for op in self.operations)


PostProcessOperationType = typing.Union[CustomPostProcessOperation, ModePower, Sum]

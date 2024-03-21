"""Defines charge distributions"""
from __future__ import annotations

from abc import ABC
from typing import Union, Tuple

import pydantic.v1 as pd

from ..heat.viz import plot_params_heat_source

from ..base import cached_property
from ..base_sim.source import AbstractSource
from ..data.data_array import TimeDataArray

from ..viz import PlotParams

from ...constants import PERCMCUBE


class ChargeDistribution(AbstractSource, ABC):
    """Abstract charge distribution."""

    structures: Tuple[str, ...] = pd.Field(
        title="Target Structures",
        description="Names of structures where the charge distribution is applied.",
    )

    # TODO: DEAL WITH THIS
    @cached_property
    def plot_params(self) -> PlotParams:
        """Default parameters for plotting a Source object."""
        return plot_params_heat_source


class UniformChargeSource(ChargeDistribution):
    """Uniform charge distribution.

    Example
    -------
    >>> heat_source = UniformHeatSource(rate=1, structures=["box"])
    """

    charge_density: Union[float, TimeDataArray] = pd.Field(
        title="Uniform charge density",
        description="Uniform charge density applied to the structures defined in the object.",
        units=PERCMCUBE,
    )


ChargeDistributionType = Union[UniformChargeSource]

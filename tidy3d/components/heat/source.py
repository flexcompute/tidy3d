"""Defines heat material specifications"""

from __future__ import annotations

from abc import ABC
from typing import Tuple, Union

import pydantic.v1 as pd

from ...constants import VOLUMETRIC_HEAT_RATE
from ...exceptions import SetupError
from ..base import cached_property
from ..base_sim.source import AbstractSource
from ..data.data_array import TimeDataArray
from ..viz import PlotParams
from .viz import plot_params_heat_source


class HeatSource(AbstractSource, ABC):
    """Abstract heat source."""

    structures: Tuple[str, ...] = pd.Field(
        title="Target Structures",
        description="Names of structures where to apply heat source.",
    )

    @cached_property
    def plot_params(self) -> PlotParams:
        """Default parameters for plotting a Source object."""
        return plot_params_heat_source

    @pd.validator("structures", always=True)
    def check_non_empty_structures(cls, val):
        """Error if source doesn't point at any structures."""
        if len(val) == 0:
            raise SetupError("List of structures for heat source is empty.")

        return val


class UniformHeatSource(HeatSource):
    """Volumetric heat source.

    Example
    -------
    >>> heat_source = UniformHeatSource(rate=1, structures=["box"])
    """

    rate: Union[float, TimeDataArray] = pd.Field(
        title="Volumetric Heat Rate",
        description="Volumetric rate of heating or cooling (if negative) in units of "
        f"{VOLUMETRIC_HEAT_RATE}.",
        units=VOLUMETRIC_HEAT_RATE,
    )


HeatSourceType = Union[UniformHeatSource]

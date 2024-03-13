"""Defines heat material specifications"""
from __future__ import annotations

from abc import ABC
from typing import Union, Tuple

import pydantic.v1 as pd

from .viz import plot_params_heat_source

from ..base import cached_property
from ..base_sim.source import AbstractSource
from ..viz import PlotParams

from ...constants import VOLUMETRIC_HEAT_RATE


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


class UniformHeatSource(HeatSource):
    """Volumetric heat source.

    Example
    -------
    >>> heat_source = UniformHeatSource(rate=1, structures=["box"])

    See Also
    --------

    **Notebooks:**
        * `Heat Solver <../../notebooks/HeatSolver.html>`_: Basic simulation flow.
    """

    rate: Union[float] = pd.Field(
        title="Volumetric Heat Rate",
        description="Volumetric rate of heating or cooling (if negative) in units of "
        f"{VOLUMETRIC_HEAT_RATE}.",
        units=VOLUMETRIC_HEAT_RATE,
    )


HeatSourceType = Union[UniformHeatSource]

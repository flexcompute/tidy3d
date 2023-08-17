"""Defines heat material specifications"""
from __future__ import annotations

from abc import ABC
from typing import Union, Tuple

import pydantic as pd

from .viz import plot_params_heat_source

from ..base import Tidy3dBaseModel, cached_property
from ..data.data_array import TimeDataArray
from ..viz import PlotParams

from ...constants import VOLUMETRIC_HEAT_RATE


class HeatSource(ABC, Tidy3dBaseModel):
    """Abstract heat source."""

    structures: Tuple[str] = pd.Field(
        title="Target Structures",
        description=f"Names of structures where to apply heat source.",
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
    """

    rate: Union[float, TimeDataArray] = pd.Field(
        title="Volumetric Heat Rate",
        description=f"Volumetric rate of heating or cooling (if negative) in units of {VOLUMETRIC_HEAT_RATE}.",
        units=VOLUMETRIC_HEAT_RATE,
    )


HeatSourceType = Union[UniformHeatSource]

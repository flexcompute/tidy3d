"""Defines heat-charge material specifications for 'HeatChargeSimulation'"""

from __future__ import annotations

from abc import ABC
from typing import Tuple

import pydantic.v1 as pd

from tidy3d.components.base import cached_property
from tidy3d.components.base_sim.source import AbstractSource
from tidy3d.components.tcad.viz import plot_params_heat_source
from tidy3d.components.viz import PlotParams
from tidy3d.exceptions import SetupError


class AbstractHeatChargeSource(AbstractSource, ABC):
    """Abstract source for heat-charge simulations. All source types
    for 'HeatChargeSimulation' derive from this class."""

    @cached_property
    def plot_params(self) -> PlotParams:
        """Default parameters for plotting a Source object."""
        return plot_params_heat_source


class StructureBasedHeatChargeSource(AbstractHeatChargeSource):
    """Abstract class associated with structures. Sources associated
    to structures must derive from this class"""

    structures: Tuple[str, ...] = pd.Field(
        title="Target Structures",
        description="Names of structures where to apply heat source.",
    )

    @pd.validator("structures", always=True)
    def check_non_empty_structures(cls, val):
        """Error if source doesn't point at any structures."""
        if len(val) == 0:
            raise SetupError("List of structures for heat source is empty.")

        return val


class GlobalHeatChargeSource(AbstractHeatChargeSource):
    """Abstract heat/charge source applied to all structures in the simulation"""

"""Defines heat-charge material specifications for 'HeatChargeSimulation'"""

from __future__ import annotations

from abc import ABC
from typing import Tuple, Union

import pydantic.v1 as pd

from ...constants import VOLUMETRIC_HEAT_RATE
from ...exceptions import SetupError
from ...log import log
from ..base import cached_property
from ..base_sim.source import AbstractSource
from ..viz import PlotParams
from .viz import plot_params_heat_source


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


class HeatSource(StructureBasedHeatChargeSource):
    """Adds a volumetric heat source (heat sink if negative values
    are provided) to specific structures in the scene.

    Example
    -------
    >>> heat_source = HeatSource(rate=1, structures=["box"])
    """

    rate: Union[float] = pd.Field(
        title="Volumetric Heat Rate",
        description="Volumetric rate of heating or cooling (if negative) in units of "
        f"{VOLUMETRIC_HEAT_RATE}.",
        units=VOLUMETRIC_HEAT_RATE,
    )


class HeatFromElectricSource(GlobalHeatChargeSource):
    """Volumetric heat source generated from an electric simulation.
    If a `HeatFromElectricSource` is specified as a source, appropriate boundary
    conditions for an electric simulation must be provided, since such a simulation
    will be executed before the heat simulation can run.

    Example
    -------
    >>> heat_source = HeatFromElectricSource()
    """


class UniformHeatSource(HeatSource):
    """Volumetric heat source. This class is deprecated. You can use
    'HeatSource' instead.

    Example
    -------
    >>> heat_source = UniformHeatSource(rate=1, structures=["box"]) # doctest: +SKIP
    """

    # NOTE: wrapper for backwards compatibility.

    @pd.root_validator(skip_on_failure=True)
    def issue_warning_deprecated(cls, values):
        """Issue warning for 'UniformHeatSource'."""
        log.warning(
            "'UniformHeatSource' is deprecated and will be discontinued. You can use "
            "'HeatSource' instead."
        )
        return values


HeatChargeSourceType = Union[HeatSource, HeatFromElectricSource, UniformHeatSource]

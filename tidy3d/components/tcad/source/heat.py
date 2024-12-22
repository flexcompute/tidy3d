"""Defines heat-charge material specifications for 'HeatChargeSimulation'"""

from __future__ import annotations

from typing import Union

import pydantic.v1 as pd

from tidy3d.components.tcad.source.abstract import StructureBasedHeatChargeSource
from tidy3d.constants import VOLUMETRIC_HEAT_RATE
from tidy3d.log import log


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

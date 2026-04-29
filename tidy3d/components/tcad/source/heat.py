"""Defines heat-charge material specifications for 'HeatChargeSimulation'"""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from tidy3d.components.data.data_array import SpatialDataArray
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

    rate: float | SpatialDataArray = Field(
        title="Volumetric Heat Rate",
        description="Volumetric rate of heating or cooling (if negative).",
        json_schema_extra={"units": VOLUMETRIC_HEAT_RATE},
    )


class UniformHeatSource(HeatSource):
    """Volumetric heat source. This class is deprecated. You can use
    :class:`HeatSource` instead.

    Example
    -------
    >>> heat_source = UniformHeatSource(rate=1, structures=["box"]) # doctest: +SKIP
    """

    # NOTE: wrapper for backwards compatibility.

    @model_validator(mode="before")
    @classmethod
    def issue_warning_deprecated(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Issue warning for 'UniformHeatSource'."""
        log.warning(
            "'UniformHeatSource' is deprecated and will be discontinued. You can use "
            "'HeatSource' instead."
        )
        return data

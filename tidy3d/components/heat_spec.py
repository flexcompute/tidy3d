"""Defines heat material specifications"""
from __future__ import annotations

from abc import ABC

import pydantic.v1 as pd

from .types import Union
from .base import Tidy3dBaseModel
from ..constants import SPECIFIC_HEAT_CAPACITY, THERMAL_CONDUCTIVITY


# Liquid class
class AbstractHeatSpec(ABC, Tidy3dBaseModel):
    """Abstract heat material specification."""


class FluidSpec(AbstractHeatSpec):
    """Fluid medium.

    Example
    -------
    >>> solid = FluidSpec()
    """


class SolidSpec(AbstractHeatSpec):
    """Solid medium.

    Example
    -------
    >>> solid = SolidSpec(
    ...     capacity=2,
    ...     conductivity=3,
    ... )
    """

    capacity: pd.PositiveFloat = pd.Field(
        title="Heat capacity",
        description=f"Volumetric heat capacity in unit of {SPECIFIC_HEAT_CAPACITY}.",
        units=SPECIFIC_HEAT_CAPACITY,
    )

    conductivity: pd.PositiveFloat = pd.Field(
        title="Thermal conductivity",
        description=f"Thermal conductivity of material in units of {THERMAL_CONDUCTIVITY}.",
        units=THERMAL_CONDUCTIVITY,
    )


HeatSpecType = Union[FluidSpec, SolidSpec]

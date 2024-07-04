"""Defines heat material specifications"""

from __future__ import annotations

from abc import ABC

import pydantic.v1 as pd

from ..constants import CONDUCTIVITY, SPECIFIC_HEAT_CAPACITY, THERMAL_CONDUCTIVITY
from .base import Tidy3dBaseModel
from .types import Union


# Liquid class
class AbstractHeatChargeSpec(ABC, Tidy3dBaseModel):
    """Abstract heat material specification."""


class FluidSpec(AbstractHeatChargeSpec):
    """Fluid medium. Heat simulations will not solve for temperature
    in a structure that has a medium with this 'heat_spec'.

    Example
    -------
    >>> solid = FluidSpec()
    """


class SolidSpec(AbstractHeatChargeSpec):
    """Solid medium for heat simulations.

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


class InsulatorSpec(AbstractHeatChargeSpec):
    """Insulating medium. Conduction simulations will not solve for electric
    potential in a structure that has a medium with this 'electric_spec'.

    Example
    -------
    >>> solid = InsulatorSpec()
    """


class ConductorSpec(AbstractHeatChargeSpec):
    """Conductor medium for conduction simulations.

    Example
    -------
    >>> solid = ConductorSpec(conductivity=3)
    """

    conductivity: pd.PositiveFloat = pd.Field(
        title="Electric conductivity",
        description=f"Electric conductivity of material in units of {CONDUCTIVITY}.",
        units=CONDUCTIVITY,
    )


ThermalSpecType = Union[FluidSpec, SolidSpec]
ElectricSpecType = Union[InsulatorSpec, ConductorSpec]

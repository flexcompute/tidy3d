"""Defines heat material specifications"""

from __future__ import annotations

from abc import ABC

import pydantic.v1 as pd

from ..constants import CONDUCTIVITY, PERMITTIVITY, SPECIFIC_HEAT_CAPACITY, THERMAL_CONDUCTIVITY
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


class ChargeSpec(AbstractHeatChargeSpec):
    """Abstract class for Charge specifications"""

    permittivity: float = pd.Field(
        1.0, ge=1.0, title="Permittivity", description="Relative permittivity.", units=PERMITTIVITY
    )


class InsulatorSpec(ChargeSpec):
    """Insulating medium. Conduction simulations will not solve for electric
    potential in a structure that has a medium with this 'electric_spec'.

    Example
    -------
    >>> solid = InsulatingSpec()
    >>> solid2 = InsulatingSpec(permittivity=1.1)

    Note: relative permittivity will be assumed 1 if no value is specified.
    """


class ConductorSpec(ChargeSpec):
    """Conductor medium for conduction simulations.

    Example
    -------
    >>> solid = ConductorSpec(conductivity=3)

    Note: relative permittivity will be assumed 1 if no value is specified.
    """

    conductivity: pd.PositiveFloat = pd.Field(
        title="Electric conductivity",
        description=f"Electric conductivity of material in units of {CONDUCTIVITY}.",
        units=CONDUCTIVITY,
    )


ThermalSpecType = Union[FluidSpec, SolidSpec]
ElectricSpecType = Union[InsulatorSpec, ConductorSpec]

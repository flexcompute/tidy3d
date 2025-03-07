"""Defines heat material specifications"""

from __future__ import annotations

from abc import ABC
from typing import Union

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import (
    SPECIFIC_HEAT_CAPACITY,
    THERMAL_CONDUCTIVITY,
)


# Liquid class
class AbstractHeatMedium(ABC, Tidy3dBaseModel):
    """Abstract heat material specification."""

    name: str = pd.Field(None, title="Name", description="Optional unique name for medium.")

    @property
    def heat(self):
        """
        This means that a heat medium has been defined inherently within this solver medium.
        This provides interconnection with the `MultiPhysicsMedium` higher-dimensional classes.
        """
        return self

    @property
    def charge(self):
        return ValueError(f"A `charge` medium does not exist in this Medium definition: {self}")

    @property
    def electrical(self):
        return ValueError(
            f"An `electrical` medium does not exist in this Medium definition: {self}"
        )

    @property
    def optical(self):
        return ValueError(f"An `optical` medium does not exist in this Medium definition: {self}")


class FluidMedium(AbstractHeatMedium):
    """Fluid medium. Heat simulations will not solve for temperature
    in a structure that has a medium with this 'heat_spec'.

    Example
    -------
    >>> solid = FluidMedium()
    """


class FluidSpec(FluidMedium):
    """Fluid medium class for backwards compatibility"""


class SolidMedium(AbstractHeatMedium):
    """Solid medium for heat simulations.

    Example
    -------
    >>> solid = SolidMedium(
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


class SolidSpec(SolidMedium):
    """Solid medium class for backwards compatibility"""


ThermalSpecType = Union[FluidSpec, SolidSpec, SolidMedium, FluidMedium]
# Note this needs to remain here to avoid circular imports in the new medium structure.

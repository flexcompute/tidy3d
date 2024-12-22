"""Defines heat material specifications"""

from __future__ import annotations

from abc import ABC

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import (
    SPECIFIC_HEAT_CAPACITY,
    THERMAL_CONDUCTIVITY,
)


# Liquid class
class AbstractHeatMedium(ABC, Tidy3dBaseModel):
    """Abstract heat material specification."""

    @property
    def heat(self):
        """
        This means that a heat medium has been defined inherently within this solver medium.
        This provides interconnection with the `MultiPhysicsMedium` higher-dimensional classes.
        """
        return self

    @property
    def optical(self):
        # Needs to be included for backwards compatibility.
        return self


class FluidSpec(AbstractHeatMedium):
    """Fluid medium. Heat simulations will not solve for temperature
    in a structure that has a medium with this 'heat_spec'.

    Example
    -------
    >>> solid = FluidSpec()
    """


class SolidSpec(AbstractHeatMedium):
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

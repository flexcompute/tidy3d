"""Defines heat material specifications"""

from __future__ import annotations

from abc import ABC
from typing import Union

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import (
    DENSITY,
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
        raise ValueError(f"A `charge` medium does not exist in this Medium definition: {self}")

    @property
    def electrical(self):
        raise ValueError(f"An `electrical` medium does not exist in this Medium definition: {self}")

    @property
    def optical(self):
        raise ValueError(f"An `optical` medium does not exist in this Medium definition: {self}")


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
        None,
        title="Heat capacity",
        description=f"Specific heat capacity in unit of {SPECIFIC_HEAT_CAPACITY}.",
        units=SPECIFIC_HEAT_CAPACITY,
    )

    conductivity: pd.PositiveFloat = pd.Field(
        title="Thermal conductivity",
        description=f"Thermal conductivity of material in units of {THERMAL_CONDUCTIVITY}.",
        units=THERMAL_CONDUCTIVITY,
    )

    density: pd.PositiveFloat = pd.Field(
        None,
        title="Density",
        description=f"Mass density of material in units of {DENSITY}.",
        units=DENSITY,
    )

    def from_si_units(
        conductivity: pd.PositiveFloat,
        capacity: pd.PositiveFloat = None,
        density: pd.PositiveFloat = None,
    ):
        """Create a SolidMedium using SI units"""
        new_conductivity = conductivity * 1e-6  # Convert from W/(m*K) to W/(um*K)
        new_capacity = capacity
        new_density = density

        if density is not None:
            new_density = density * 1e-18

        return SolidMedium(
            capacity=new_capacity,
            conductivity=new_conductivity,
            density=new_density,
        )


class SolidSpec(SolidMedium):
    """Solid medium class for backwards compatibility"""


ThermalSpecType = Union[FluidSpec, SolidSpec, SolidMedium, FluidMedium]
# Note this needs to remain here to avoid circular imports in the new medium structure.

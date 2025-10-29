"""Defines heat material specifications"""

from __future__ import annotations

from abc import ABC
from typing import Union

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.constants import (
    DENSITY,
    DYNAMIC_VISCOSITY,
    SPECIFIC_HEAT,
    SPECIFIC_HEAT_CAPACITY,
    THERMAL_CONDUCTIVITY,
    THERMAL_EXPANSIVITY,
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
    def charge(self) -> None:
        raise ValueError(f"A `charge` medium does not exist in this Medium definition: {self}")

    @property
    def electrical(self) -> None:
        raise ValueError(f"An `electrical` medium does not exist in this Medium definition: {self}")

    @property
    def optical(self) -> None:
        raise ValueError(f"An `optical` medium does not exist in this Medium definition: {self}")


class FluidMedium(AbstractHeatMedium):
    """Fluid medium. Heat simulations will not solve for temperature
    in a structure that has a medium with this ``heat_spec``.


    Notes
    --------
    The full set of parameters is primarily intended for calculations involving natural
    convection, where they are used to determine the heat transfer coefficient.
    In the current version, these specific properties may not be utilized for
    other boundary condition types.

    Examples
    --------
    >>> # If you are using a boundary condition without a natural convection model,
    >>> # the specific properties of the fluid are not required. In this common
    >>> # scenario, you can instantiate the class without arguments.
    >>> air = FluidMedium()

    >>> # It is most convenient to define the fluid from standard SI units
    >>> # using the `from_si_units` classmethod.
    >>> # The following defines air at approximately 20°C.
    >>> air_from_si = FluidMedium.from_si_units(
    ...     thermal_conductivity=0.0257,  # Unit: W/(m*K)
    ...     viscosity=1.81e-5,          # Unit: Pa*s
    ...     specific_heat=1005,         # Unit: J/(kg*K)
    ...     density=1.204,              # Unit: kg/m^3
    ...     expansivity=1/293.15        # Unit: 1/K
    ... )

    >>> # One can also define the medium directly in Tidy3D units.
    >>> # The following is equivalent to the example above.
    >>> air_direct = FluidMedium(
    ...     thermal_conductivity=2.57e-8,
    ...     viscosity=1.81e-11,
    ...     specific_heat=1.005e+15,
    ...     density=1.204e-18,
    ...     expansivity=1/293.15
    ... )
    """

    thermal_conductivity: pd.NonNegativeFloat = pd.Field(
        default=None,
        title="Fluid Thermal Conductivity",
        description="Thermal conductivity (k) of the fluid.",
        units=THERMAL_CONDUCTIVITY,
    )
    viscosity: pd.NonNegativeFloat = pd.Field(
        default=None,
        title="Fluid Dynamic Viscosity",
        description="Dynamic viscosity (μ) of the fluid.",
        units=DYNAMIC_VISCOSITY,
    )
    specific_heat: pd.NonNegativeFloat = pd.Field(
        default=None,
        title="Fluid Specific Heat",
        description="Specific heat of the fluid at constant pressure.",
        units=SPECIFIC_HEAT,
    )
    density: pd.NonNegativeFloat = pd.Field(
        default=None,
        title="Fluid Density",
        description="Density (ρ) of the fluid.",
        units=DENSITY,
    )
    expansivity: pd.NonNegativeFloat = pd.Field(
        default=None,
        title="Fluid Thermal Expansivity",
        description="Thermal expansion coefficient (β) of the fluid.",
        units=THERMAL_EXPANSIVITY,
    )

    def from_si_units(
        thermal_conductivity: pd.NonNegativeFloat,
        viscosity: pd.NonNegativeFloat,
        specific_heat: pd.NonNegativeFloat,
        density: pd.NonNegativeFloat,
        expansivity: pd.NonNegativeFloat,
    ):
        thermal_conductivity_tidy = thermal_conductivity / 1e6  # W/(m*K) -> W/(um*K)
        viscosity_tidy = viscosity / 1e6  # Pa*s -> kg/(um*s)
        specific_heat_tidy = specific_heat * 1e12  # J/(kg*K) -> um**2/(s**2*K)
        density_tidy = density / 1e18  # kg/m**3 -> kg/um**3
        expansivity_tidy = expansivity  # 1/K -> 1/K (no change)

        return FluidMedium(
            thermal_conductivity=thermal_conductivity_tidy,
            viscosity=viscosity_tidy,
            specific_heat=specific_heat_tidy,
            density=density_tidy,
            expansivity=expansivity_tidy,
        )


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

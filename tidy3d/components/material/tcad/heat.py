"""Defines heat material specifications"""

from __future__ import annotations

from abc import ABC
from math import isfinite
from typing import TYPE_CHECKING

from pydantic import Field, NonNegativeFloat, PositiveFloat, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import Coordinate
from tidy3d.constants import (
    DENSITY,
    DYNAMIC_VISCOSITY,
    SPECIFIC_HEAT,
    SPECIFIC_HEAT_CAPACITY,
    THERMAL_CONDUCTIVITY,
    THERMAL_EXPANSIVITY,
    VELOCITY,
)

if TYPE_CHECKING:
    from tidy3d.compat import Self


# Liquid class
class AbstractHeatMedium(ABC, Tidy3dBaseModel):
    """Abstract heat material specification."""

    name: str | None = Field(None, title="Name", description="Optional unique name for medium.")

    @property
    def heat(self) -> Self:
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

    thermal_conductivity: NonNegativeFloat | None = Field(
        default=None,
        title="Fluid Thermal Conductivity",
        description="Thermal conductivity (k) of the fluid.",
        json_schema_extra={"units": THERMAL_CONDUCTIVITY},
    )
    viscosity: NonNegativeFloat | None = Field(
        default=None,
        title="Fluid Dynamic Viscosity",
        description="Dynamic viscosity (μ) of the fluid.",
        json_schema_extra={"units": DYNAMIC_VISCOSITY},
    )
    specific_heat: NonNegativeFloat | None = Field(
        default=None,
        title="Fluid Specific Heat",
        description="Specific heat of the fluid at constant pressure.",
        json_schema_extra={"units": SPECIFIC_HEAT},
    )
    density: NonNegativeFloat | None = Field(
        default=None,
        title="Fluid Density",
        description="Density (ρ) of the fluid.",
        json_schema_extra={"units": DENSITY},
    )
    expansivity: NonNegativeFloat | None = Field(
        default=None,
        title="Fluid Thermal Expansivity",
        description="Thermal expansion coefficient (β) of the fluid.",
        json_schema_extra={"units": THERMAL_EXPANSIVITY},
    )

    @classmethod
    def from_si_units(
        cls,
        thermal_conductivity: NonNegativeFloat,
        viscosity: NonNegativeFloat,
        specific_heat: NonNegativeFloat,
        density: NonNegativeFloat,
        expansivity: NonNegativeFloat,
    ) -> Self:
        thermal_conductivity_tidy = thermal_conductivity / 1e6  # W/(m*K) -> W/(um*K)
        viscosity_tidy = viscosity / 1e6  # Pa*s -> kg/(um*s)
        specific_heat_tidy = specific_heat * 1e12  # J/(kg*K) -> um**2/(s**2*K)
        density_tidy = density / 1e18  # kg/m**3 -> kg/um**3
        expansivity_tidy = expansivity  # 1/K -> 1/K (no change)

        return cls(
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

    capacity: PositiveFloat | None = Field(
        None,
        title="Heat capacity",
        description=f"Specific heat capacity in unit of {SPECIFIC_HEAT_CAPACITY}.",
        json_schema_extra={"units": SPECIFIC_HEAT_CAPACITY},
    )

    conductivity: PositiveFloat = Field(
        title="Thermal conductivity",
        description=f"Thermal conductivity of material in units of {THERMAL_CONDUCTIVITY}.",
        json_schema_extra={"units": THERMAL_CONDUCTIVITY},
    )

    density: PositiveFloat | None = Field(
        None,
        title="Density",
        description=f"Mass density of material in units of {DENSITY}.",
        json_schema_extra={"units": DENSITY},
    )

    velocity: Coordinate | None = Field(
        None,
        title="Advection velocity",
        description="Constant advection velocity ``(vx, vy, vz)`` of the solid medium in units "
        f"of {VELOCITY}. When set to a nonzero value, the heat solver adds a convective "
        "transport term ``rho * cp * V . grad(T)`` for structures using this medium; both "
        "``capacity`` and ``density`` must then be provided, since the ``rho * cp`` coefficient "
        "is ``capacity * density``. Leave as ``None`` (the default) for pure conduction. Note: a "
        "velocity with a nonzero component normal to an interface between two touching solids is "
        "only supported when ``rho * cp * V . n`` matches across the face; a normal-flux jump at "
        "such an interface is ill-posed and unsupported (velocities tangential to the interface "
        "are fine). This term is applied by the heat solver, including when heat is coupled "
        "with electrical conduction. It is not supported in non-isothermal charge "
        "(coupled charge+heat) simulations, where the coupled thermal solve does not apply "
        "it, so setting a nonzero ``velocity`` there raises a setup error.",
        json_schema_extra={"units": VELOCITY},
    )

    @model_validator(mode="after")
    def _check_velocity_requires_capacity_and_density(self) -> Self:
        """Velocity components must be finite, since the solver consumes them as a
        real advection speed. A nonzero ``velocity`` also enables convective transport,
        whose ``rho * cp`` coefficient is ``capacity * density``; require both so the
        solver receives a positive coefficient instead of an unset-property sentinel."""
        if self.velocity is not None:
            if any(not isfinite(v) for v in self.velocity):
                self._raise_validation_error_at_loc(
                    "'velocity' components must be finite numbers.",
                    "velocity",
                )
            if any(v != 0.0 for v in self.velocity) and (
                self.capacity is None or self.density is None
            ):
                self._raise_validation_error_at_loc(
                    "A nonzero 'velocity' enables convective heat transport, which requires "
                    "both 'capacity' and 'density' to be set (the convection coefficient is "
                    "'capacity * density'). Please provide both.",
                    "velocity",
                )
        return self

    @classmethod
    def from_si_units(
        cls,
        conductivity: PositiveFloat,
        capacity: PositiveFloat | None = None,
        density: PositiveFloat | None = None,
        velocity: Coordinate | None = None,
    ) -> Self:
        """Create a SolidMedium using SI units"""
        new_conductivity = conductivity * 1e-6  # Convert from W/(m*K) to W/(um*K)
        new_capacity = capacity
        new_density = density
        new_velocity = velocity

        if density is not None:
            new_density = density * 1e-18

        if velocity is not None:
            new_velocity = tuple(v * 1e6 for v in velocity)  # Convert from m/s to um/s

        return cls(
            capacity=new_capacity,
            conductivity=new_conductivity,
            density=new_density,
            velocity=new_velocity,
        )


class SolidSpec(SolidMedium):
    """Solid medium class for backwards compatibility"""


ThermalSpecType = FluidSpec | SolidSpec | SolidMedium | FluidMedium
# Note this needs to remain here to avoid circular imports in the new medium structure.

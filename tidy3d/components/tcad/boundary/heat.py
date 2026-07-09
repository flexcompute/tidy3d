"""Defines heat material specifications"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import Field, NonNegativeFloat, PositiveFloat, field_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.material.tcad.heat import FluidMedium
from tidy3d.components.tcad.boundary.abstract import HeatChargeBC
from tidy3d.constants import (
    ACCELERATION,
    GRAV_ACC,
    HEAT_FLUX,
    HEAT_TRANSFER_COEFF,
    KELVIN,
    MICROMETER,
    THERMAL_RESISTANCE,
)

if TYPE_CHECKING:
    from tidy3d.compat import Self


class TemperatureBC(HeatChargeBC):
    """Constant temperature thermal boundary conditions.

    Example
    -------
    >>> import tidy3d as td
    >>> bc = td.TemperatureBC(temperature=300)
    """

    temperature: PositiveFloat = Field(
        title="Temperature",
        description="Temperature value.",
        json_schema_extra={"units": KELVIN},
    )


class HeatFluxBC(HeatChargeBC):
    """Constant flux thermal boundary conditions.

    Example
    -------
    >>> import tidy3d as td
    >>> bc = td.HeatFluxBC(flux=1)
    """

    flux: float = Field(
        title="Heat Flux",
        description="Heat flux value.",
        json_schema_extra={"units": HEAT_FLUX},
    )


class VerticalNaturalConvectionCoeffModel(Tidy3dBaseModel):
    """
    Specification for natural convection from a vertical plate.

    Notes
    -----

    This class calculates the heat transfer coefficient (h) based on fluid
    properties and an expected temperature difference, then provides these
    values as  :math:`\\text{base_l}`,  :math:`\\text{base_nl}`, and  :math:`\\text{exponent}`  for a generalized heat flux equation

    .. math::

        q = \\text{base_nl} * (T_\\text{surf} - T_\\text{fluid})^\\text{exponent} + \\text{base}_{l} * (T_\\text{surf}- T_\\text{fluid}).

    """

    medium: FluidMedium | None = Field(
        default=None,
        title="Interface medium",
        description=(
            "The :class:`FluidMedium` used for the heat transfer coefficient calculation. "
            "If `None`, the fluid is automatically deduced from the interface, which can be defined"
            "by either a :class:`MediumMediumInterface` or a :class:`StructureStructureInterface`."
        ),
    )

    plate_length: NonNegativeFloat = Field(
        title="Plate Characteristic Length",
        description="Characteristic length (L), defined as the height of the vertical plate.",
        json_schema_extra={"units": MICROMETER},
    )

    gravity: NonNegativeFloat = Field(
        default=GRAV_ACC,
        title="Gravitational Acceleration",
        description="Gravitational acceleration (g).",
        json_schema_extra={"units": ACCELERATION},
    )

    @classmethod
    def from_si_units(
        cls,
        plate_length: NonNegativeFloat,
        medium: FluidMedium = None,
        gravity: NonNegativeFloat = GRAV_ACC * 1e-6,
    ) -> Self:
        """
        Create an instance from standard SI units.

        Args:
            plate_length: Plate characteristic length in [m].
            gravity: Gravitational acceleration in [m/s**2].

        Returns:
            An instance of VerticalNaturalConvectionCoeffModel with all values
            converted to Tidy3D's internal unit system.
        """

        # --- Apply conversion factors ---
        # value_tidy = value_si * factor
        plate_length_tidy = plate_length * 1e6  # m -> um
        g_tidy = gravity * 1e6  # m/s**2 -> um/s**2

        return cls(
            medium=medium,
            plate_length=plate_length_tidy,
            gravity=g_tidy,
        )


class ConvectionBC(HeatChargeBC):
    """Convective thermal boundary conditions.

    Example
    -------
    >>> import tidy3d as td
    >>> bc = td.ConvectionBC(ambient_temperature=300, transfer_coeff=1)

    >>> # Convection with a natural convection model.
    >>> # First, define the fluid medium (e.g. air at 300 K).
    >>> air = td.FluidMedium.from_si_units(
    ...     thermal_conductivity=0.0257,  # Unit: W/(m*K)
    ...     viscosity=1.81e-5,          # Unit: Pa*s
    ...     specific_heat=1005,         # Unit: J/(kg*K)
    ...     density=1.204,              # Unit: kg/m^3
    ...     expansivity=1/293.15        # Unit: 1/K
    ... )
    >>>
    >>> # Next, create the model, which requires the fluid and a characteristic length.
    >>> natural_conv_model = td.VerticalNaturalConvectionCoeffModel.from_si_units(
    ...     medium=air, plate_length=1e-5
    ... )
    >>>
    >>> # Finally, create the boundary condition using this model.
    >>> bc_natural = td.ConvectionBC(
    ...     ambient_temperature=300, transfer_coeff=natural_conv_model
    ... )

    >>> # If the fluid medium is not provided to the coefficient model, it is automatically retrieved from
    >>> # the interface.
    >>> natural_conv_model = td.VerticalNaturalConvectionCoeffModel.from_si_units(plate_length=1e-5)
    >>> bc_natural_nom = td.ConvectionBC(
    ...     ambient_temperature=300, transfer_coeff=natural_conv_model
    ... )
    """

    ambient_temperature: PositiveFloat = Field(
        title="Ambient Temperature",
        description="Ambient temperature.",
        json_schema_extra={"units": KELVIN},
    )

    transfer_coeff: NonNegativeFloat | VerticalNaturalConvectionCoeffModel = Field(
        title="Heat Transfer Coefficient",
        description="Heat transfer coefficient value.",
        json_schema_extra={"units": HEAT_TRANSFER_COEFF},
    )

    emissivity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Surface Emissivity",
        description=(
            "Optional surface emissivity. When set to a positive value, a gray-body "
            "radiative exchange term is added to the convective flux, using the same "
            "ambient temperature. An emissivity of 0 adds no radiative flux and is "
            "equivalent to leaving the field unset. "
            "See :class:`RadiationBC` for the definition of the radiative term and "
            "the simulation types that support it."
        ),
    )


class RadiationBC(HeatChargeBC):
    """Gray-body surface radiation thermal boundary condition.

    Notes
    -----

    The boundary radiates to (and absorbs from) a far-field ambient at
    ``ambient_temperature`` following the Stefan-Boltzmann law

    .. math::

        q = \\varepsilon \\sigma \\left( T^4 - T_{amb}^4 \\right),

    where :math:`\\varepsilon` is the surface emissivity and :math:`\\sigma` the
    Stefan-Boltzmann constant. This boundary condition is applied by the heat solver
    (heat and conduction+heat simulations). It is not yet supported in non-isothermal
    charge (coupled charge+heat) simulations, which reject it at validation.

    Example
    -------
    >>> import tidy3d as td
    >>> bc = td.RadiationBC(ambient_temperature=300, emissivity=0.9)
    """

    ambient_temperature: PositiveFloat = Field(
        title="Ambient Temperature",
        description="Far-field ambient temperature the surface radiates to.",
        json_schema_extra={"units": KELVIN},
    )

    emissivity: float = Field(
        gt=0.0,
        le=1.0,
        title="Surface Emissivity",
        description=(
            "Surface emissivity (dimensionless, between 0 and 1). Must be strictly "
            "positive: a zero-emissivity surface exchanges no radiative flux, so the "
            "boundary condition would not constrain the temperature."
        ),
    )


class ThermalContactResistance(HeatChargeBC):
    """Interfacial thermal resistance (thermal contact / Kapitza resistance) between two
    touching solids.

    Notes
    -----

    The interface transmits the heat flux

    .. math::

        q'' = \\frac{1}{R} \\left( T_1 - T_2 \\right),

    so the temperature is discontinuous across the interface: the temperature jump is
    proportional to the heat flux crossing it. This models thin thermally-resistive
    interfaces (imperfect bonding, grain boundaries, phonon mismatch between thin films)
    without meshing them. This condition can only be placed on an interface between two
    solids (:class:`StructureStructureInterface` or :class:`MediumMediumInterface`).

    This boundary condition is applied by the heat solver, including when heat is coupled
    with electrical conduction. It is not supported in non-isothermal charge (coupled
    charge+heat) simulations, where the coupled thermal solve does not apply the
    interfacial thermal resistance, so including it there raises a setup error.

    Example
    -------
    >>> import tidy3d as td
    >>> bc = td.ThermalContactResistance(resistance=3e3)  # K*um^2/W
    >>> bc_si = td.ThermalContactResistance.from_si_units(resistance=3e-9)  # m^2*K/W
    """

    resistance: PositiveFloat = Field(
        title="Interfacial Thermal Resistance",
        description=f"Interfacial thermal resistance in units of {THERMAL_RESISTANCE}.",
        json_schema_extra={"units": THERMAL_RESISTANCE},
    )

    @field_validator("resistance")
    @classmethod
    def _resistance_must_be_finite(cls, val: float) -> float:
        """Reject non-finite resistance values."""
        if not math.isfinite(val):
            raise ValueError(
                "'resistance' must be finite. To thermally decouple the two sides of an "
                "interface, remove one of them from the heat simulation instead."
            )
        return val

    @classmethod
    def from_si_units(cls, resistance: PositiveFloat) -> Self:
        """Create a :class:`ThermalContactResistance` using SI units.

        Args:
            resistance: Interfacial thermal resistance in [m^2*K/W].

        Returns:
            An instance of ThermalContactResistance with the value converted to Tidy3D's
            internal unit system.
        """
        resistance_tidy = resistance * 1e12  # m^2*K/W -> K*um^2/W
        return cls(resistance=resistance_tidy)

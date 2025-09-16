"""Defines heat material specifications"""

from __future__ import annotations

from typing import Union

import pydantic.v1 as pd

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
)


class TemperatureBC(HeatChargeBC):
    """Constant temperature thermal boundary conditions.

    Example
    -------
    >>> import tidy3d as td
    >>> bc = td.TemperatureBC(temperature=300)
    """

    temperature: pd.PositiveFloat = pd.Field(
        title="Temperature",
        description="Temperature value.",
        units=KELVIN,
    )


class HeatFluxBC(HeatChargeBC):
    """Constant flux thermal boundary conditions.

    Example
    -------
    >>> import tidy3d as td
    >>> bc = td.HeatFluxBC(flux=1)
    """

    flux: float = pd.Field(
        title="Heat Flux",
        description="Heat flux value.",
        units=HEAT_FLUX,
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

    medium: FluidMedium = pd.Field(
        default=None,
        title="Interface medium",
        description=(
            "The :class:`FluidMedium` used for the heat transfer coefficient calculation. "
            "If `None`, the fluid is automatically deduced from the interface, which can be defined"
            "by either a :class:`MediumMediumInterface` or a :class:`StructureStructureInterface`."
        ),
    )

    plate_length: pd.NonNegativeFloat = pd.Field(
        title="Plate Characteristic Length",
        description="Characteristic length (L), defined as the height of the vertical plate.",
        units=MICROMETER,
    )

    gravity: pd.NonNegativeFloat = pd.Field(
        default=GRAV_ACC,
        title="Gravitational Acceleration",
        description="Gravitational acceleration (g).",
        units=ACCELERATION,
    )

    def from_si_units(
        plate_length: pd.NonNegativeFloat,
        medium: FluidMedium = None,
        gravity: pd.NonNegativeFloat = GRAV_ACC * 1e-6,
    ):
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

        return VerticalNaturalConvectionCoeffModel(
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

    ambient_temperature: pd.PositiveFloat = pd.Field(
        title="Ambient Temperature",
        description="Ambient temperature.",
        units=KELVIN,
    )

    transfer_coeff: Union[pd.NonNegativeFloat, VerticalNaturalConvectionCoeffModel] = pd.Field(
        title="Heat Transfer Coefficient",
        description="Heat transfer coefficient value.",
        units=HEAT_TRANSFER_COEFF,
    )

"""Defines heat material specifications"""

from __future__ import annotations

from abc import ABC

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel

from ..constants import (
    SPECIFIC_HEAT_CAPACITY,
    THERMAL_CONDUCTIVITY,
)
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


#
# class ChargeMedium(AbstractHeatChargeSpec):
#     """Abstract class for Charge specifications"""
#
# # permittivity: float = pd.Field(
# #     1.0, ge=1.0, title="Permittivity", description="Relative permittivity.", units=PERMITTIVITY
# # )
# #
# # class ChargeMedium(ChargeMedium):
# #     """Insulating medium. Conduction simulations will not solve for electric
# #     potential in a structure that has a medium with this 'electric_spec'.
# #
# #     Example
# #     -------
# #     >>> solid = InsulatingSpec()
# #     >>> solid2 = InsulatingSpec(permittivity=1.1)
# #
# #     Note: relative permittivity will be assumed 1 if no value is specified.
# #     """
# #
# #
# # class ChargeMedium(ChargeMedium):
# #     """Conductor medium for conduction simulations.
# #
# #     Example
# #     -------
# #     >>> solid = ChargeMedium(conductivity=3)
# #
# #     Note: relative permittivity will be assumed 1 if no value is specified.
# #     """
# #
# #     conductivity: pd.PositiveFloat = pd.Field(
# #         1,
# #         title="Electric conductivity",
# #         description=f"Electric conductivity of material in units of {CONDUCTIVITY}.",
# #         units=CONDUCTIVITY,
# #     )
#
# class ActiveSemiConductorMedium(ChargeMedium):
#     """
#     This class is used to define semiconductors.
#
#     Notes
#     -----
#         Both acceptors and donors can be either a positive number or an 'xarray.DataArray'.
#         Default values for parameters and models are those appropriate for Silicon
#     """
#
#     nc: pd.PositiveFloat = pd.Field(
#         2.86e19,
#         title="Effective density of electron states",
#         description="Effective density of electron states",
#         units="cm^(-3)",
#     )
#
#     nv: pd.PositiveFloat = pd.Field(
#         3.1e19,
#         title="Effective density of hole states",
#         description="Effective density of hole states",
#         units="cm^(-3)",
#     )
#
#     eg: pd.PositiveFloat = pd.Field(
#         1.11,
#         title="Band-gap energy",
#         description="Band-gap energy",
#         units=ELECTRON_VOLT,
#     )
#
#     chi: float = pd.Field(
#         4.05, title="Electron affinity", description="Electron affinity", units=ELECTRON_VOLT
#     )
#
#     mobility_model: MobilityModelTypes = pd.Field(
#         CaugheyThomasMobility(),
#         title="Mobility model",
#         description="Mobility model",
#     )
#
#     recombination_model: Tuple[RecombinationModelTypes, ...] = pd.Field(
#         (ShockleyReedHallRecombination(), AugerRecombination(), RadiativeRecombination()),
#         title="Recombination models",
#         description="Array containing the recombination models to be applied to the material.",
#     )
#
#     bandgap_model: BandGapModelTypes = pd.Field(
#         SlotboomNarrowingBandGap(),
#         title="Bandgap narrowing model.",
#         description="Bandgap narrowing model.",
#     )
#
#     acceptors: Union[pd.NonNegativeFloat, SpatialDataArray] = pd.Field(
#         0,
#         title="Doping: Acceptor concentration",
#         description="Units of 1/cm^3",
#         units="1/cm^3",
#     )
#
#     donors: Union[pd.NonNegativeFloat, SpatialDataArray] = pd.Field(
#         0,
#         title="Doping: Donor concentration",
#         description="Units of 1/cm^3",
#         units="1/cm^3",
#     )


ThermalSpecType = Union[FluidSpec, SolidSpec]
# ElectricSpecType = Union[ChargeMedium, ActiveSemiConductorMedium]

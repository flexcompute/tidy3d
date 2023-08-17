"""Defines heat material specifications"""
from __future__ import annotations

from abc import ABC
from typing import Union

import pydantic as pd

from ..base import Tidy3dBaseModel
from ..bc_placement import BCPlacementType

from ...constants import KELVIN, HEAT_FLUX, HEAT_TRANSFER_COEFF


class HeatBC(ABC, Tidy3dBaseModel):
    """Abstract thermal boundary conditions."""


class TemperatureBC(HeatBC):
    """Constant temperature thermal boundary conditions."""

    temperature: pd.PositiveFloat = pd.Field(
        title="Temperature",
        description=f"Temperature value in units of {KELVIN}.",
        units=KELVIN,
    )


class HeatFluxBC(HeatBC):
    """Constant flux thermal boundary conditions."""

    flux: float = pd.Field(
        title="Heat Flux",
        description=f"Heat flux value in units of {HEAT_FLUX}.",
        units=HEAT_FLUX,
    )


class ConvectionBC(HeatBC):
    """Convective thermal boundary conditions."""

    ambient_temperature: pd.PositiveFloat = pd.Field(
        title="Ambient Temperature",
        description=f"Ambient temperature value in units of {KELVIN}.",
        units=KELVIN,
    )

    transfer_coeff: pd.NonNegativeFloat = pd.Field(
        title="Heat Transfer Coefficient",
        description=f"Heat flux value in units of {HEAT_TRANSFER_COEFF}.",
        units=HEAT_TRANSFER_COEFF,
    )


class HeatBCInterface(HeatBC):
    """Temperature and flux continuity."""


HeatBCType = Union[TemperatureBC, HeatFluxBC, ConvectionBC, HeatBCInterface]


class HeatBoundarySpec(Tidy3dBaseModel):
    """Heat boundary conditions specification."""

    placement: BCPlacementType = pd.Field(
        title="Boundary Conditions Placement",
        description="Location to apply boundary conditions.",
    )

    condition: HeatBCType = pd.Field(
        title="Boundary Conditions",
        description="Boundary conditions to apply at the selected location.",
    )

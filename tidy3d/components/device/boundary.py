"""Defines heat material specifications"""
from __future__ import annotations

from abc import ABC
from typing import Union

import pydantic.v1 as pd

from ..base import Tidy3dBaseModel
from ..bc_placement import BCPlacementType
from ..types import TYPE_TAG_STR

from ...constants import KELVIN, HEAT_FLUX, HEAT_TRANSFER_COEFF
from ...constants import VOLT, CURRENT_DENSITY


class DeviceBC(ABC, Tidy3dBaseModel):
    """Abstract device boundary conditions."""


class TemperatureBC(DeviceBC):
    """Constant temperature thermal boundary conditions.

    Example
    -------
    >>> bc = TemperatureBC(temperature=300)
    """

    temperature: pd.PositiveFloat = pd.Field(
        title="Temperature",
        description=f"Temperature value in units of {KELVIN}.",
        units=KELVIN,
    )


class HeatFluxBC(DeviceBC):
    """Constant flux thermal boundary conditions.

    Example
    -------
    >>> bc = HeatFluxBC(flux=1)
    """

    flux: float = pd.Field(
        title="Heat Flux",
        description=f"Heat flux value in units of {HEAT_FLUX}.",
        units=HEAT_FLUX,
    )


class ConvectionBC(DeviceBC):
    """Convective thermal boundary conditions.

    Example
    -------
    >>> bc = ConvectionBC(ambient_temperature=300, transfer_coeff=1)
    """

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


class VoltageBC(DeviceBC):
    """Electric potential (voltage) boundary condition.
    Sets a potential at the specified boundary.

    Example
    -------
    >>> bc = VoltageBC(potential=2)
    """

    voltage: pd.FiniteFloat = pd.Field(
        title="Voltage",
        description="Electric potential to be applied at the specified boundary.",
        units=VOLT,
    )


class CurrentBC(DeviceBC):
    """Current boundary conditions.

    Example
    -------
    >>> bc = CurrentBC(current_density=1)
    """

    current_density: pd.FiniteFloat = pd.Field(
        title="Current density",
        description=f"Current density in units of {CURRENT_DENSITY}.",
        units=CURRENT_DENSITY,
    )


class InsulatingBC(DeviceBC):
    """Insulation boundary condition.
    Ensures electric fields as well as the surface recombination current density
    are set to zero.

    Example
    -------
    >>> bc = InsulatingBC()
    """


DeviceBoundaryConditionType = Union[
    TemperatureBC, HeatFluxBC, ConvectionBC, VoltageBC, CurrentBC, InsulatingBC
]


class DeviceBoundarySpec(Tidy3dBaseModel):
    """Device boundary conditions specification.

    Example
    -------
    >>> from tidy3d import SimulationBoundary
    >>> bc_spec = HeatBoundarySpec(
    ...     placement=SimulationBoundary(),
    ...     condition=ConvectionBC(ambient_temperature=300, transfer_coeff=1),
    ... )
    """

    placement: BCPlacementType = pd.Field(
        title="Boundary Conditions Placement",
        description="Location to apply boundary conditions.",
        discriminator=TYPE_TAG_STR,
    )

    condition: DeviceBoundaryConditionType = pd.Field(
        title="Boundary Conditions",
        description="Boundary conditions to apply at the selected location.",
        discriminator=TYPE_TAG_STR,
    )


HeatBoundarySpec = Union[DeviceBoundarySpec]
"""Heat BC specification
NOTE: here for backward-compatibility only."""

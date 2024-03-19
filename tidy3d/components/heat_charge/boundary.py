"""Defines heat material specifications"""

from __future__ import annotations

from abc import ABC
from typing import Tuple, Union

import pydantic.v1 as pd

from ...constants import CURRENT_DENSITY, HEAT_FLUX, HEAT_TRANSFER_COEFF, KELVIN, VOLT
from ..base import Tidy3dBaseModel
from ..bc_placement import BCPlacementType
from ..types import TYPE_TAG_STR


class HeatChargeBC(ABC, Tidy3dBaseModel):
    """Abstract heat-charge boundary conditions."""


class TemperatureBC(HeatChargeBC):
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


class HeatFluxBC(HeatChargeBC):
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


class ConvectionBC(HeatChargeBC):
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


class VoltageBC(HeatChargeBC):
    """Electric potential (voltage) boundary condition.
    Sets a potential at the specified boundary.

    Example
    -------
    >>> bc = VoltageBC(voltage=2)
    """

    voltage: Union[pd.FiniteFloat, Tuple[pd.FiniteFloat, ...]] = pd.Field(
        title="Voltage",
        description="Electric potential to be applied at the specified boundary.",
        units=VOLT,
    )


class CurrentBC(HeatChargeBC):
    """Current boundary conditions.

    Example
    -------
    >>> bc = CurrentBC(current_density=1)
    """

    current_density: pd.FiniteFloat = pd.Field(
        title="Current density",
        description="Current density.",
        units=CURRENT_DENSITY,
    )


class InsulatingBC(HeatChargeBC):
    """Insulation boundary condition.
    Ensures electric fields as well as the surface recombination current density
    are set to zero.

    Example
    -------
    >>> bc = InsulatingBC()
    """


HeatChargeBoundaryConditionType = Union[
    TemperatureBC, HeatFluxBC, ConvectionBC, VoltageBC, CurrentBC, InsulatingBC
]


class HeatChargeBoundarySpec(Tidy3dBaseModel):
    """Heat-Charge boundary conditions specification.

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

    condition: HeatChargeBoundaryConditionType = pd.Field(
        title="Boundary Conditions",
        description="Boundary conditions to apply at the selected location.",
        discriminator=TYPE_TAG_STR,
    )


class HeatBoundarySpec(HeatChargeBoundarySpec):
    """Heat BC specification
    NOTE: here for backward-compatibility only."""

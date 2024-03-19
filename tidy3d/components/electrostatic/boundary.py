"""Defines heat material specifications"""
from __future__ import annotations

from abc import ABC
from typing import Union

import pydantic.v1 as pd

from ..base import Tidy3dBaseModel
from ..bc_placement import BCPlacementType

#from ...constants import KELVIN, HEAT_FLUX, HEAT_TRANSFER_COEFF


class ElectricBC(ABC, Tidy3dBaseModel):
    """Abstract electric boundary conditions."""


class PotentialBC(ElectricBC):
    """Potential boundary condition.
    Sets a potential at the specified boundary.

    Example
    -------
    >>> bc = PotentialBC(potential=2)
    """

    potential: pd.PositiveFloat = pd.Field(
        title="Potential",
        description="Electric potential to be applied at the specified boundary.",
        units="Volt",
    )


class InsulatingBC(ElectricBC):
    """Insulation boundary condition.
    Ensures electric fields as well as the surface recombination current density
    are set to zero.

    Example
    -------
    >>> bc = InsulatingBC()
    """

    # flux: float = pd.Field(
    #     title="Heat Flux",
    #     description=f"Heat flux value in units of {HEAT_FLUX}.",
    #     units=HEAT_FLUX,
    # )


ElectricBoundaryConditionType = Union[PotentialBC, InsulatingBC]


class ElectricBoundarySpec(Tidy3dBaseModel):
    """Electric boundary conditions specification.

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
    )

    condition: ElectricBoundaryConditionType = pd.Field(
        title="Boundary Conditions",
        description="Boundary conditions to apply at the selected location.",
    )

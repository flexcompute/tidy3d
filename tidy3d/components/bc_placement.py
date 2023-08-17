"""Defines heat material specifications"""
from __future__ import annotations

from abc import ABC
from typing import Union, Tuple

import pydantic as pd

from .base import Tidy3dBaseModel
from ..exceptions import SetupError


class AbstractBCPlacement(ABC, Tidy3dBaseModel):
    """Abstract placement for boundary conditions."""


class StructureBoundary(AbstractBCPlacement):
    """Placement of boundary conditions on the structure's boundary.

    Example
    -------
    >>> bc_placement = StructureBoundary(structure="box")
    """

    structure: str = pd.Field(
        title="Structure Name",
        description="Name of the structure.",
    )


class StructureStructureInterface(AbstractBCPlacement):
    """Placement of boundary conditions between two structures.

    Example
    -------
    >>> bc_placement = StructureStructureInterface(structures=["box", "sphere"])
    """

    structures: Tuple[str, str] = pd.Field(
        title="Structures",
        description="Names of two structures.",
    )

    @pd.validator("structures", always=True)
    def unique_names(cls, val):
        """Error if the same structure is provided twice"""
        if val[0] == val[1]:
            raise SetupError(
                "The same structure is provided twice in :class:.`StructureStructureInterface`."
            )
        return val


class MediumMediumInterface(AbstractBCPlacement):
    """Placement of boundary conditions between two mediums.

    Example
    -------
    >>> bc_placement = MediumMediumInterface(mediums=["dieletric", "metal"])
    """

    mediums: Tuple[str, str] = pd.Field(
        title="Mediums",
        description="Names of two mediums.",
    )

    @pd.validator("mediums", always=True)
    def unique_names(cls, val):
        """Error if the same structure is provided twice"""
        if val[0] == val[1]:
            raise SetupError(
                "The same medium is provided twice in :class:.`MediumMediumInterface`."
            )
        return val


class SimulationBoundary(AbstractBCPlacement):
    """Placement of boundary conditions on the simulation box boundary.

    Example
    -------
    >>> bc_placement = SimulationBoundary()
    """


class StructureSimulationBoundary(AbstractBCPlacement):
    """Placement of boundary conditions on the simulation box boundary covered by the structure.

    Example
    -------
    >>> bc_placement = StructureSimulationBoundary(structure="box")
    """

    structure: str = pd.Field(
        title="Structure Name",
        description="Name of the structure.",
    )


BCPlacementType = Union[
    StructureBoundary,
    StructureStructureInterface,
    MediumMediumInterface,
    SimulationBoundary,
    StructureSimulationBoundary,
]

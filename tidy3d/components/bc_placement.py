"""Defines placements for boundary conditions."""

from __future__ import annotations

from abc import ABC
from typing import Union

from pydantic import Field, field_validator

from tidy3d.components.types.base import discriminated_union
from tidy3d.exceptions import SetupError

from .base import Tidy3dBaseModel
from .types import BoxSurface


class AbstractBCPlacement(ABC, Tidy3dBaseModel):
    """Abstract placement for boundary conditions."""


class StructureBoundary(AbstractBCPlacement):
    """Placement of boundary conditions on the structure's boundary.

    Example
    -------
    >>> bc_placement = StructureBoundary(structure="box")
    """

    structure: str = Field(
        title="Structure Name",
        description="Name of the structure.",
    )


class StructureStructureInterface(AbstractBCPlacement):
    """Placement of boundary conditions between two structures.

    Example
    -------
    >>> bc_placement = StructureStructureInterface(structures=["box", "sphere"])
    """

    structures: tuple[str, str] = Field(
        title="Structures",
        description="Names of two structures.",
    )

    @field_validator("structures")
    @classmethod
    def unique_names(cls, val: tuple[str, str]) -> tuple[str, str]:
        """Error if the same structure is provided twice"""
        if val[0] == val[1]:
            raise SetupError(
                "The same structure is provided twice in 'StructureStructureInterface'."
            )
        return val


class MediumMediumInterface(AbstractBCPlacement):
    """Placement of boundary conditions between two mediums.

    Example
    -------
    >>> bc_placement = MediumMediumInterface(mediums=["dieletric", "metal"])
    """

    mediums: tuple[str, str] = Field(
        title="Mediums",
        description="Names of two mediums.",
    )

    @field_validator("mediums")
    @classmethod
    def unique_names(cls, val: tuple[str, str]) -> tuple[str, str]:
        """Error if the same structure is provided twice"""
        if val[0] == val[1]:
            raise SetupError("The same medium is provided twice in 'MediumMediumInterface'.")
        return val


class SimulationBoundary(AbstractBCPlacement):
    """Placement of boundary conditions on the simulation box boundary.

    Example
    -------
    >>> bc_placement = SimulationBoundary(surfaces=["x-", "x+"])
    """

    surfaces: tuple[BoxSurface, ...] = Field(
        ("x-", "x+", "y-", "y+", "z-", "z+"),
        title="Surfaces",
        description="Surfaces of simulation domain where to apply boundary conditions.",
    )


class StructureSimulationBoundary(AbstractBCPlacement):
    """Placement of boundary conditions on the simulation box boundary covered by the structure.

    Example
    -------
    >>> bc_placement = StructureSimulationBoundary(structure="box", surfaces=["y-", "y+"])
    """

    structure: str = Field(
        title="Structure Name",
        description="Name of the structure.",
    )

    surfaces: tuple[BoxSurface, ...] = Field(
        ("x-", "x+", "y-", "y+", "z-", "z+"),
        title="Surfaces",
        description="Surfaces of simulation domain where to apply boundary conditions.",
    )


BCPlacementType = discriminated_union(
    Union[
        StructureBoundary,
        StructureStructureInterface,
        MediumMediumInterface,
        SimulationBoundary,
        StructureSimulationBoundary,
    ]
)

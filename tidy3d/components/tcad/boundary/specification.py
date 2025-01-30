"""Defines heat material specifications"""

from __future__ import annotations

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.bc_placement import BCPlacementType
from tidy3d.components.tcad.types import HeatChargeBCType
from tidy3d.components.types import TYPE_TAG_STR


class HeatChargeBoundarySpec(Tidy3dBaseModel):
    """Heat-Charge boundary conditions specification.

    Example
    -------
    >>> import tidy3d as td
    >>> bc_v1 = td.HeatChargeBoundarySpec(
    ...   condition=td.VoltageBC(source=td.DCVoltageSource(voltage=0)),
    ...   placement=td.StructureBoundary(structure="contact_left"),
    ... )
    """

    placement: BCPlacementType = pd.Field(
        title="Boundary Conditions Placement",
        description="Location to apply boundary conditions.",
        discriminator=TYPE_TAG_STR,
    )

    condition: HeatChargeBCType = pd.Field(
        title="Boundary Conditions",
        description="Boundary conditions to apply at the selected location.",
        discriminator=TYPE_TAG_STR,
    )


class HeatBoundarySpec(HeatChargeBoundarySpec):
    """Heat BC specification. DEPRECIATED.

    Warning
    -------
        Included backward-compatibility only.

    Example
    --------
    >>> import tidy3d as td
    >>> bc_spec = td.HeatBoundarySpec(
    ...     placement=td.SimulationBoundary(),
    ...     condition=td.ConvectionBC(ambient_temperature=300, transfer_coeff=1),
    ... )

    """

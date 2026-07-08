"""Defines heat material specifications"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from tidy3d.components.spice.sources.types import CurrentSourceType, VoltageSourceType
from tidy3d.components.tcad.boundary.abstract import HeatChargeBC
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.constants import CURRENT_DENSITY, VOLT

ContactModelType = Literal["ohmic", "schottky_mott"]


class VoltageBC(HeatChargeBC):
    """
    Constant electric potential (voltage) :math:`= \\text{V}` boundary condition.
    Sets a potential at the specified boundary.

    Notes
    -----

        In charge simulations it also accepts an array of voltages.
        In this case, a solution for each of these voltages will
        be computed.

        A Schottky contact can be enabled by setting ``model="schottky_mott"``,
        which uses the Schottky-Mott rule together with Richardson-Dushman
        thermionic emission. The default ``model="ohmic"`` keeps the ohmic
        contact behavior.

        Place a Schottky contact on the metal structure's
        :class:`.StructureBoundary` (the contact may span semiconductor and
        insulator faces of the metal, e.g. under an oxide cladding) or on a
        :class:`.StructureStructureInterface` between the metal and the
        semiconductor. A single Schottky contact must touch exactly one
        semiconductor medium.

    Example
    -------
    >>> import tidy3d as td
    >>> voltage_source = td.DCVoltageSource(voltage=1)
    >>> voltage_bc = td.VoltageBC(source=voltage_source)
    """

    source: VoltageSourceType = Field(
        discriminator=TYPE_TAG_STR,
        title="Voltage",
        description="Electric potential to be applied at the specified boundary.",
        json_schema_extra={"units": VOLT},
    )

    model: ContactModelType = Field(
        "ohmic",
        title="Contact model",
        description='Contact model. ``"ohmic"`` (default) is the ohmic '
        'contact path. ``"schottky_mott"`` enables the Schottky-Mott + '
        "Richardson-Dushman thermionic contact and requires ``work_function`` "
        "on the adjacent :class:`.ChargeConductorMedium` plus "
        "``electron_affinity``, ``richardson_electron``, ``richardson_hole`` on "
        "the adjacent :class:`.SemiconductorMedium`.",
    )


class CurrentBC(HeatChargeBC):
    """
    Current boundary conditions.

    Example
    -------
    >>> import tidy3d as td
    >>> current_source = td.DCCurrentSource(current=1)
    >>> current_bc = CurrentBC(source=current_source)
    """

    source: CurrentSourceType = Field(
        title="Current Source",
        description="A current source",
        json_schema_extra={"units": CURRENT_DENSITY},
    )
    # TODO translation between currentsource amps and currentdensity, why not amps here?


class InsulatingBC(HeatChargeBC):
    """Insulation boundary condition.

    Notes
    -----

        Ensures the electric potential to the normal :math:`\\nabla \\psi \\cdot \\mathbf{n}  = 0` as well as the
        surface recombination current density :math:`J_s = \\mathbf{J} \\cdot \\mathbf{n} = 0` are set to zero where
        the current density is :math:`\\mathbf{J}` and the normal vector is :math:`\\mathbf{n}`

    Example
    -------
    >>> bc = InsulatingBC()
    """

"""Defines heat material specifications"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from tidy3d.components.spice.sources.types import CurrentSourceType, VoltageSourceType
from tidy3d.components.tcad.boundary.abstract import HeatChargeBC
from tidy3d.components.tcad.generation_recombination import SurfaceRecombinationModelType
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


class SurfaceRecombinationBC(HeatChargeBC):
    """Surface recombination at a semiconductor boundary or zone interface.

    Notes
    -----

        Adds a Robin-type recombination flux to the carrier continuity equations,

        .. math::

           \\mathbf{J}_n \\cdot \\hat{\\mathbf{n}} = -q\\,R_s(n,p),
           \\qquad
           \\mathbf{J}_p \\cdot \\hat{\\mathbf{n}} = +q\\,R_s(n,p),

        The kinetic kernel :math:`R_s(n,p)` is supplied by ``model``. A
        separate :class:`VoltageBC` may share the same contact face; in that
        composition the contact pins the potential and this BC supplies the
        finite surface-recombination carrier exchange.

    Example
    -------

        >>> import tidy3d as td
        >>> sr_bc = td.SurfaceRecombinationBC(
        ...     model=td.SurfaceShockleyReedHallRecombination(S_n=1e3, S_p=1e3),
        ... )
    """

    model: SurfaceRecombinationModelType = Field(
        ...,
        title="Surface recombination model",
        description="Kinetic recombination model.",
    )

    Q_f: float = Field(
        0.0,
        title="Fixed interface sheet charge density",
        description="Signed fixed interface sheet charge density "
        "[C/cm^2]. Negative values (e.g. for Al2O3 on Si) attract holes "
        "and deplete electrons at the interface (field-effect "
        "passivation); positive values do the inverse.",
        json_schema_extra={"units": "C/cm^2"},
    )

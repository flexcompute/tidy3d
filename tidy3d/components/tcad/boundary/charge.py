"""Defines heat material specifications"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from tidy3d.components.spice.sources.types import CurrentSourceType, VoltageSourceType
from tidy3d.components.tcad.boundary.abstract import HeatChargeBC
from tidy3d.components.tcad.generation_recombination import SurfaceRecombinationModelType
from tidy3d.components.types import TYPE_TAG_STR
from tidy3d.constants import AMP, VOLT

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

        A contact whose faces all lie on an insulator -- a gate over an
        oxide -- takes its potential reference from the metal work function
        :math:`W`. Place it on the gate metal's
        :class:`.StructureBoundary` or on the
        :class:`.StructureStructureInterface` between the metal and the
        insulator, with ``work_function`` set on the metal's
        :class:`.ChargeConductorMedium`. The contacted face must adjoin a metal
        carrying that property. The potential is then
        referenced to the vacuum level, so every :class:`.SemiconductorMedium`
        in the simulation must also carry ``electron_affinity`` -- without it
        the bands sit at an arbitrary offset from the gate and the flat-band
        voltage is wrong by whole volts. Reported potentials are measured
        against the electron affinity :math:`\\chi_\\mathrm{ref}` of the
        reference semiconductor, so a monitor reads
        :math:`V - (W - \\chi_\\mathrm{ref})` on the gate face. The reference
        semiconductor is the background ``medium`` when it is a
        :class:`.SemiconductorMedium`, and otherwise the first semiconductor
        structure in priority order, which is the lowest-priority one (with no
        ``priority`` set, the first one listed in ``structures``); with one
        semiconductor material it is
        simply that material's :math:`\\chi`. Any ``VoltageBC`` with a face on
        an insulator -- a gate, or an ohmic or Schottky contact whose metal is
        clad by an insulator -- is supported only by the accelerated charge
        solver, because only it references such a face to the metal work
        function.

        Every ``VoltageBC`` made on one :class:`.ChargeConductorMedium`
        structure is one terminal: the sweep, the AC drive and the reported
        terminal current cover all of its faces together, across zones and
        across several ``boundary_spec`` entries. Two entries naming one metal
        must therefore agree on their source.

        Contact faces that meet at a point must prescribe the same electric
        potential there. Touching terminals must also have compatible
        small-signal drives; separate their boundaries to bias them independently.

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
        default="ohmic",
        title="Contact model",
        description='Contact model. ``"ohmic"`` (default) is the ohmic '
        'contact path. ``"schottky_mott"`` enables the Schottky-Mott + '
        "Richardson-Dushman thermionic contact and requires ``work_function`` "
        "on the adjacent :class:`.ChargeConductorMedium` plus "
        "``electron_affinity``, ``richardson_electron``, ``richardson_hole`` on "
        "the adjacent :class:`.SemiconductorMedium`.",
    )


class CurrentBC(HeatChargeBC):
    """Drive a contact with a prescribed total current and solve its potential.

    Current is positive into the device and, in 2D, is per micron of device
    depth. With ``symmetry`` on the simulation, the accelerated charge solver
    takes it as the current of the whole device, not of the simulated portion.
    All faces of one :class:`.ChargeConductorMedium` form one terminal.
    Several terminals may be current-driven. At least one :class:`.VoltageBC`
    must provide the potential reference, and terminals joined through
    semiconductor regions must include a voltage-driven one. The exception is a
    terminal that is the only contact on its semiconductor regions: with
    ``current=0`` it floats, and its potential is then set by where the
    simulation starts rather than by the device. A floating terminal must be
    the only current-driven terminal of the simulation and is not supported in
    a small-signal analysis. A nonzero current needs a semiconductor path to
    another contact that crosses no :class:`.InsulatingBC`, since no steady
    state can carry it otherwise. A contact reaching only an insulator is a
    gate and is rejected, including at ``current=0``; drive a gate with
    :class:`.VoltageBC`. Supported only by the accelerated charge solver; not
    supported for a conduction simulation, where :class:`.VoltageBC` should be
    used instead.

    The contact is ohmic by default. ``model="schottky_mott"`` makes it a
    Schottky contact, with the same requirements as on :class:`.VoltageBC`. A
    :class:`.SurfaceRecombinationBC` may share the placement of an ohmic
    current-driven contact, as it may share that of a :class:`.VoltageBC`. An
    ohmic current-driven contact without a :class:`.SurfaceRecombinationBC`
    may reach a heterojunction interface; a Schottky contact or one sharing its
    placement with a :class:`.SurfaceRecombinationBC` may not.

    An array of currents sweeps the terminal. The currents are solved in the
    order of a voltage sweep: zero first, then the positive currents in
    increasing order, then the negative ones by increasing magnitude, each
    starting from the solution of the previous one. At most one
    ``boundary_spec`` entry may sweep, by a voltage array or a current array,
    and every terminal a swept ``CurrentBC`` entry names carries the swept
    current. The current-voltage data of a current sweep are indexed by the
    solved potential of a swept terminal, which need not increase with the
    current. Without a sweep, the reported current-voltage point is that of
    the first current-driven terminal in ``boundary_spec``, at its solved
    potential.

    In a small-signal analysis a current-driven terminal is an open circuit:
    its potential follows the small-signal drive while its current stays at
    the prescribed value. With a current sweep the small-signal analysis runs
    at the last entry of ``current``, and ``at_voltages`` is not supported.

    If the device's current-voltage curve passes through the requested current
    at more than one voltage (for example, a negative differential resistance
    region), the voltage this boundary condition settles on depends on where
    the simulation starts, not only on the requested current. A current sweep
    starts each point from the previous one, so closely spaced currents follow
    one branch of the curve.

    Example
    -------
    >>> import tidy3d as td
    >>> current_source = td.DCCurrentSource(current=1e-3)
    >>> current_bc = CurrentBC(source=current_source)
    """

    source: CurrentSourceType = Field(
        title="Current Source",
        description="Total current through this terminal, positive into the device and, in 2D, per micron of device depth. With ``symmetry`` on the simulation, it is the current of the whole device, not of the simulated portion.",
        json_schema_extra={"units": AMP},
    )

    model: ContactModelType = Field(
        default="ohmic",
        title="Contact model",
        description='Contact model. ``"ohmic"`` (default) is the ohmic '
        'contact path. ``"schottky_mott"`` enables the Schottky-Mott + '
        "Richardson-Dushman thermionic contact and requires ``work_function`` "
        "on the adjacent :class:`.ChargeConductorMedium` plus "
        "``electron_affinity``, ``richardson_electron``, ``richardson_hole`` on "
        "the adjacent :class:`.SemiconductorMedium`.",
    )


class InsulatingBC(HeatChargeBC):
    """Insulation boundary condition.

    Notes
    -----

        Ensures the electric potential to the normal :math:`\\nabla \\psi \\cdot \\mathbf{n}  = 0` as well as the
        surface recombination current density :math:`J_s = \\mathbf{J} \\cdot \\mathbf{n} = 0` are set to zero where
        the current density is :math:`\\mathbf{J}` and the normal vector is :math:`\\mathbf{n}`

        On a face that also carries a :class:`.VoltageBC` or :class:`.CurrentBC`, the contact
        wins whatever the order in ``boundary_spec``, unless the placement of this condition
        names the contact's metal structure or medium; then the later entry wins.

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

        The condition may also be placed on an interface between two materials
        that both take part in the charge simulation, such as a
        semiconductor/insulator interface (for example Si/SiO2), or a junction
        between two differently doped regions of the same material, as at a
        regrowth or wafer-bonded interface. There the interface keeps its
        material continuity and the recombination applies on the semiconductor
        side. An interface with no :class:`SemiconductorMedium` on either side
        is rejected, since it has no carriers to recombine.

        Free surfaces, semiconductor/insulator interfaces and same-material
        junctions take :class:`SurfaceShockleyReedHallRecombination`. A
        heterojunction (semiconductors differing in ``N_c``, ``N_v``, ``E_g`` or
        ``electron_affinity``) takes
        :class:`HeterojunctionSurfaceShockleyReedHallRecombination` instead,
        placed with a :class:`StructureStructureInterface` or a
        :class:`MediumMediumInterface` that names its two sides.

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
        discriminator=TYPE_TAG_STR,
    )

    Q_f: float = Field(
        default=0.0,
        title="Fixed interface sheet charge density",
        description="Signed fixed interface sheet charge density "
        "[C/cm^2]. Negative values (e.g. for Al2O3 on Si) attract holes "
        "and deplete electrons at the interface (field-effect "
        "passivation); positive values do the inverse.",
        json_schema_extra={"units": "C/cm^2"},
    )

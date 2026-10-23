"""Objects that define how data is recorded from simulation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from tidy3d.components.tcad.monitors.abstract import HeatChargeMonitor


class SteadyPotentialMonitor(HeatChargeMonitor):
    """
    Electric potential (:math:`\\psi`) monitor.

    Notes
    -----
    The potential is defined up to a constant. On the accelerated charge solver it is measured
    against the electron affinity of the reference semiconductor described in
    :class:`.VoltageBC`; the non-accelerated charge solver ignores electron affinity.

    Example
    -------
    >>> import tidy3d as td
    >>> voltage_monitor_z0 = td.SteadyPotentialMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="voltage_z0", unstructured=True,
    ... )
    """


class SteadyFreeCarrierMonitor(HeatChargeMonitor):
    """
    Free-carrier monitor for Charge simulations.

    Example
    -------
    >>> import tidy3d as td
    >>> carrier_monitor_z0 = td.SteadyFreeCarrierMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="carrier_z0", unstructured=True,
    ... )
    """

    # NOTE: for the time being supporting unstructured
    unstructured: Literal[True] = Field(
        default=True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )


class SteadyEnergyBandMonitor(HeatChargeMonitor):
    """
    Energy bands monitor for Charge simulations.

    Example
    -------
    >>> import tidy3d as td
    >>> energy_monitor_z0 = td.SteadyEnergyBandMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="bands_z0", unstructured=True,
    ... )
    """

    # NOTE: for the time being supporting unstructured
    unstructured: Literal[True] = Field(
        default=True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )


class SteadyCapacitanceMonitor(HeatChargeMonitor):
    """
    Capacitance monitor associated with a charge simulation.

    Example
    -------
    >>> import tidy3d as td
    >>> capacitance_global_mnt = td.SteadyCapacitanceMonitor(
    ... center=(0, 0.14, 0), size=(td.inf, td.inf, 0), name="capacitance_global_mnt",
    ... )
    """

    # NOTE: for the time being supporting unstructured
    unstructured: Literal[True] = Field(
        default=True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )


class SteadyElectricFieldMonitor(HeatChargeMonitor):
    """
    Electric field monitor for Charge/Conduction simulations.

    Example
    -------
    >>> import tidy3d as td
    >>> electric_field_monitor_z0 = td.SteadyElectricFieldMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="electric_field_z0",
    ... )
    """

    unstructured: Literal[True] = Field(
        default=True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )


class SteadyCurrentDensityMonitor(HeatChargeMonitor):
    """
    Current density monitor for Charge/Conduction simulations.

    Example
    -------
    >>> import tidy3d as td
    >>> current_density_monitor_z0 = td.SteadyCurrentDensityMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="current_density_z0",
    ... )
    """

    unstructured: Literal[True] = Field(
        default=True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )


class SteadyGenerationRecombinationMonitor(HeatChargeMonitor):
    """
    Generation-recombination rate monitor for Charge simulations.

    Notes
    -----
    Records the per-node net generation-recombination rate entering the carrier
    continuity equations, :math:`U = R - G`, together with a per-mechanism
    breakdown. **Recombination is positive** (it removes carriers) and
    **generation is negative** (it adds carriers). ``net_recombination`` covers
    every generation and recombination mechanism active in the simulation's
    media — Shockley-Reed-Hall, Auger, radiative, band-to-band tunneling,
    distributed carrier generation, and impact ionization — and each has its own
    breakdown field. Surface recombination is a boundary flux, not part of this
    per-node volume rate. Each breakdown field is reported when its mechanism is
    active in some medium and stays ``None`` otherwise; the breakdown fields sum to
    ``net_recombination``. Rates are in :math:`cm^{-3}\\,s^{-1}`.
    Available only on the accelerated solver.

    Example
    -------
    >>> import tidy3d as td
    >>> generation_monitor = td.SteadyGenerationRecombinationMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="generation_z0",
    ... )
    """

    unstructured: Literal[True] = Field(
        default=True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )


class SteadyDopingMonitor(HeatChargeMonitor):
    """
    Doping monitor for Charge simulations.

    Notes
    -----
    Records the total donor and acceptor concentrations :math:`N_d` and :math:`N_a` in
    :math:`cm^{-3}`, which do not depend on bias. When some semiconductor medium sets
    ``donor_ionization`` or ``acceptor_ionization``, it also records the ionized
    concentrations :math:`N_d^+` and :math:`N_a^-` that enter the Poisson equation at each
    bias point, which follow the local quasi-Fermi levels; where a medium has no law they
    equal the doping. Without any law in the simulation the ionized fields are ``None``.
    The ionized fraction is one division, for example :math:`N_d^+/N_d` where :math:`N_d > 0`.
    Available only on the accelerated solver.

    Example
    -------
    >>> import tidy3d as td
    >>> doping_monitor = td.SteadyDopingMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="doping_z0",
    ... )
    """

    unstructured: Literal[True] = Field(
        default=True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )


class SelfHeatingMonitor(HeatChargeMonitor):
    """
    Volumetric self-heating monitor for Charge/Conduction simulations.

    Notes
    -----
    Records the total volumetric heat generation rate :math:`q` (:math:`W/\\mu m^3`)
    produced by the electrical solve. In a ``Conduction`` simulation this is Joule
    heating :math:`\\vec{J} \\cdot \\vec{E}`; in a ``Charge`` simulation it is the sum
    of Joule and recombination heating. The two contributions are not reported
    separately.

    The recorded field can be fed back into a ``Heat`` simulation as a
    :class:`HeatSource` via ``SelfHeatingData.to_spatial_data_array``. This monitor
    counts as a Charge simulation's required output monitor, so it can be the only one.

    In a ``Conduction`` simulation, :class:`HeatFromElectricSource` can drive a heat
    solve directly instead, without this monitor. That source is *not* available from an
    isothermal Charge simulation, which solves at a list of bias points a single heat
    solve cannot consume; for a self-consistent charge-heat solution use a non-isothermal
    analysis spec (:class:`SteadyChargeDCAnalysis`), which couples both physics internally
    and ignores the source.

    Example
    -------
    >>> import tidy3d as td
    >>> self_heating_monitor = td.SelfHeatingMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="self_heat",
    ... )
    """

    unstructured: Literal[True] = Field(
        default=True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )


class SteadyChargeResidualMonitor(HeatChargeMonitor):
    """
    Per-node residual monitor for Charge simulations (debug tool).

    Notes
    -----
    Records the per-node signed residual of each governing equation:
    :math:`R_\\psi` (Poisson), :math:`R_n` (electron continuity), :math:`R_p`
    (hole continuity), and :math:`R_T` (heat) when the thermal solver is active.
    The electron and hole continuity equations express carrier conservation.
    The values are dimensionless and on the same scale as the simulation's
    convergence tolerance, so the nodes with the largest :math:`|R|` (those
    approaching or exceeding that tolerance) are where the solution least
    satisfies the equations (the least-converged regions). Available only
    through the accelerated solver.

    Example
    -------
    >>> import tidy3d as td
    >>> residual_monitor = td.SteadyChargeResidualMonitor(
    ... center=(0, 0.14, 0), size=(0.6, 0.3, 0), name="residual_z0",
    ... )
    """

    unstructured: Literal[True] = Field(
        default=True,
        title="Unstructured Grid",
        description="Return data on the original unstructured grid.",
    )

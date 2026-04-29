"""EME monitors"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from pydantic import Field, NonNegativeInt, PositiveInt

from tidy3d.components.base_sim.monitor import AbstractMonitor
from tidy3d.components.monitor import (
    AbstractFieldMonitor,
    MediumMonitor,
    ModeSolverMonitor,
    PermittivityMonitor,
)
from tidy3d.components.types import FreqArray

if TYPE_CHECKING:
    from .sweep import EMESweepSpecType

BYTES_COMPLEX = 8


class EMEMonitor(AbstractMonitor, ABC):
    """Abstract base class for EME monitors.

    Notes
    -----
        EME monitors record data from the EME eigenmode expansion and propagation.
        Unlike FDTD monitors, they do not record time-domain data; instead they
        record modal quantities such as eigenmodes, mode coefficients, and
        fields reconstructed from the EME basis.
    """

    freqs: FreqArray | None = Field(
        None,
        title="Monitor Frequencies",
        description="Frequencies at which the monitor will record. "
        "Must be a subset of the simulation 'freqs'. "
        "A value of 'None' will record at all simulation 'freqs'.",
    )

    num_modes: NonNegativeInt | None = Field(
        None,
        title="Number of Modes",
        description="Maximum number of modes for the monitor to record. "
        "Cannot exceed the greatest number of modes in any EME cell. "
        "A value of 'None' will record all modes.",
    )

    num_sweep: NonNegativeInt | None = Field(
        1,
        title="Number of Sweep Indices",
        description="Number of sweep indices for the monitor to record. "
        "Cannot exceed the number of sweep indices for the simulation. "
        "If the sweep does not change the monitor data, the sweep index "
        "will be omitted. A value of 'None' will record all sweep indices.",
    )

    interval_space: tuple[Literal[1], Literal[1], Literal[1]] = Field(
        (1, 1, 1),
        title="Spatial Interval",
        description="Number of grid step intervals between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last point of the monitor grid are always included. "
        "Not all monitors support values different from 1.",
    )

    eme_cell_interval_space: Literal[1] = Field(
        1,
        title="EME Cell Interval",
        description="Number of EME cells between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last cells are always included. Not used in all monitors. "
        "Not all monitors support values different from 1.",
    )

    colocate: Literal[True] = Field(
        True,
        title="Colocate Fields",
        description="Defines whether fields are colocated to grid cell boundaries (i.e. to the "
        "primal grid) on-the-fly during a solver run. Can be toggled for field recording monitors "
        "and is hard-coded for other monitors depending on their specific function.",
    )

    @abstractmethod
    def storage_size(
        self,
        num_cells: int,
        num_transverse_cells: int,
        num_eme_cells: int,
        num_virtual_eme_cells: int,
        num_freqs: int,
        num_modes: int,
        sweep_spec: EMESweepSpecType | None,
    ) -> int:
        """Size of monitor storage given the number of points after discretization.

        Parameters
        ----------
        num_cells : int
            Number of grid cells within the monitor after discretization by a :class:`.Simulation`.
        num_transverse_cells: int
            Number of grid cells within the monitor transverse to the propagation axis
            after discretization by a :class:`.Simulation`.
        num_eme_cells: int
            Number of EME cells intersecting the monitor.
        num_virtual_eme_cells: int
            Number of virtual EME cells intersecting the monitor (includes repetitions).
        num_freqs: int
            Number of frequencies in the monitor.
        num_modes: int
            Number of modes in the monitor.
        sweep_spec: Optional[EMESweepSpecType]
            The sweep specification used in the simulation.

        Returns
        -------
        int
            Number of bytes to be stored in monitor.
        """

    def _effective_num_sweep(self, num_sweep: int) -> int:
        """Apply monitor sweep limiting if specified."""
        if self.num_sweep is None:
            return num_sweep
        return min(num_sweep, self.num_sweep)


class EMEModeSolverMonitor(EMEMonitor):
    """EME mode solver monitor.
    Records EME modes computed in planes intersecting the monitor geometry.

    Note
    ----

        This is different than a :class:`.ModeSolverMonitor`, which computes modes within
        its planar geometry. In contrast, this monitor does not compute new modes; instead,
        it records the modes used for EME expansion and propagation, but only within the
        monitor geometry.

    Example
    -------
    >>> monitor = EMEModeSolverMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     freqs=[300e12],
    ...     num_modes=2,
    ...     name="eme_modes"
    ... )
    """

    interval_space: tuple[Literal[1], Literal[1], Literal[1]] = Field(
        (1, 1, 1),
        title="Spatial Interval",
        description="Note: not yet supported. Number of grid step intervals between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last point of the monitor grid are always included. Note: the interval "
        "in the propagation direction is not used. Note: this is not yet supported.",
    )

    eme_cell_interval_space: PositiveInt = Field(
        1,
        title="EME Cell Interval",
        description="Number of EME cells between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last cells are always included. Not used in all monitors. "
        "Not all monitors support values different from 1.",
    )

    colocate: bool = Field(
        True,
        title="Colocate Fields",
        description="Toggle whether fields should be colocated to grid cell boundaries (i.e. "
        "primal grid nodes). Default (False) is used internally in EME propagation.",
    )

    normalize: bool = Field(
        True,
        title="Normalize Modes",
        description="Whether to normalize the EME modes to unity flux.",
    )

    keep_invalid_modes: bool = Field(
        False,
        title="Keep Invalid Modes",
        description="Whether to store modes containing nan values and modes which are "
        "exponentially increasing in the propagation direction.",
    )

    def storage_size(
        self,
        num_cells: int,
        num_transverse_cells: int,
        num_eme_cells: int,
        num_virtual_eme_cells: int,
        num_freqs: int,
        num_modes: int,
        sweep_spec: EMESweepSpecType | None,
    ) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # EMEModeSolverMonitor only varies with sweep for EMEFreqSweep (sweep_modes)
        num_sweep = sweep_spec.num_sweep if sweep_spec and sweep_spec.sweep_modes else 1
        num_sweep = self._effective_num_sweep(num_sweep)
        bytes_single = (
            6
            * BYTES_COMPLEX
            * num_transverse_cells
            * num_eme_cells
            * num_freqs
            * num_modes
            * num_sweep
        )
        return bytes_single


class EMEFieldMonitor(EMEMonitor, AbstractFieldMonitor):
    """EME monitor for propagated electromagnetic field.

    Notes
    -----
        Records the E and H fields assembled from EME modes and their propagation
        coefficients. The field is stored as a function of spatial coordinates,
        frequency, ``sweep_index``, ``eme_port_index``, and ``mode_index``, where
        ``eme_port_index`` indicates the excitation port and ``mode_index`` indicates
        the excited mode at that port.

    Example
    -------
    >>> monitor = EMEFieldMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,0),
    ...     freqs=[300e12],
    ...     num_modes=2,
    ...     name="eme_field"
    ... )
    """

    interval_space: tuple[PositiveInt, PositiveInt, PositiveInt] = Field(
        (1, 1, 1),
        title="Spatial Interval",
        description="Number of grid step intervals between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last point of the monitor grid are always included.",
    )

    eme_cell_interval_space: Literal[1] = Field(
        1,
        title="EME Cell Interval",
        description="Number of EME cells between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last cells are always included. Not used in all monitors. "
        "Not all monitors support values different from 1. Note: this field is not used for "
        "EME field monitor.",
    )

    colocate: bool = Field(
        True,
        title="Colocate Fields",
        description="Toggle whether fields should be colocated to grid cell boundaries (i.e. "
        "primal grid nodes). Default (False) is used internally in EME propagation.",
    )

    num_modes: NonNegativeInt | None = Field(
        None,
        title="Number of Modes",
        description="Maximum number of modes for the monitor to record. "
        "For 'EMEFieldMonitor', refers to the number of modes at each port."
        "Cannot exceed the max of the number of modes in the two ports. "
        "A value of 'None' will record all modes.",
    )

    def storage_size(
        self,
        num_cells: int,
        num_transverse_cells: int,
        num_eme_cells: int,
        num_virtual_eme_cells: int,
        num_freqs: int,
        num_modes: int,
        sweep_spec: EMESweepSpecType | None,
    ) -> int:
        """Size of monitor storage given the number of points after discretization."""
        # EMEFieldMonitor uses full sweep count
        num_sweep = sweep_spec.num_sweep if sweep_spec else 1
        num_sweep = self._effective_num_sweep(num_sweep)
        bytes_single = 6 * BYTES_COMPLEX * num_cells * num_freqs * num_modes * 2 * num_sweep
        return bytes_single


class EMECoefficientMonitor(EMEMonitor):
    """EME monitor for mode coefficients and related quantities.

    Notes
    -----
        Records the amplitudes of the forward (``A``) and backward (``B``) modes
        in each cell intersecting the monitor geometry. Additional fields can be
        recorded by including them in the ``fields`` parameter: propagation indices
        (``n_complex``), power flux (``flux``), interface S matrices
        (``interface_smatrices``), and mode overlaps (``overlaps``).

    Example
    -------
    >>> monitor = EMECoefficientMonitor(
    ...     center=(1,2,3),
    ...     size=(2,2,2),
    ...     freqs=[300e12],
    ...     num_modes=2,
    ...     fields=['A', 'B'],
    ...     name="eme_coeffs"
    ... )
    """

    fields: tuple[
        Literal["A", "B", "n_complex", "flux", "interface_smatrices", "overlaps"], ...
    ] = Field(
        ("A", "B"),
        title="Coefficient Fields",
        description="Collection of coefficient fields to store in the monitor. "
        "Available fields: 'A' (forward mode coefficients), 'B' (backward mode coefficients), "
        "'n_complex' (propagation indices), 'flux' (power flux), "
        "'interface_smatrices' (S matrices at cell interfaces), "
        "'overlaps' (mode overlaps).",
    )

    interval_space: tuple[Literal[1], Literal[1], Literal[1]] = Field(
        (1, 1, 1),
        title="Spatial Interval",
        description="Number of grid step intervals between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last point of the monitor grid are always included. "
        "Not all monitors support values different from 1. Note: This field is not used "
        "for 'EMECoefficientMonitor'.",
    )

    eme_cell_interval_space: PositiveInt = Field(
        1,
        title="EME Cell Interval",
        description="Number of EME cells between monitor recordings. If equal to 1, "
        "there will be no downsampling. If greater than 1, the step will be applied, but the "
        "first and last cells are always included. Not used in all monitors. "
        "Not all monitors support values different from 1.",
    )

    def storage_size(
        self,
        num_cells: int,
        num_transverse_cells: int,
        num_eme_cells: int,
        num_virtual_eme_cells: int,
        num_freqs: int,
        num_modes: int,
        sweep_spec: EMESweepSpecType | None,
    ) -> int:
        """Size of monitor storage given the number of points after discretization."""
        bytes_total = 0
        num_interfaces = num_eme_cells - 1

        # Compute per-field sweep counts based on sweep_spec properties
        # A and B: use full sweep count
        num_sweep_full = sweep_spec.num_sweep if sweep_spec else 1
        num_sweep_full = self._effective_num_sweep(num_sweep_full)
        # n_complex, flux, overlaps: only vary with EMEFreqSweep (sweep_modes)
        num_sweep_modes = sweep_spec.num_sweep if sweep_spec and sweep_spec.sweep_modes else 1
        num_sweep_modes = self._effective_num_sweep(num_sweep_modes)
        # interface_smatrices: only vary with EMEFreqSweep or EMEModeSweep (sweep_interfaces)
        num_sweep_interfaces = (
            sweep_spec.num_sweep if sweep_spec and sweep_spec.sweep_interfaces else 1
        )
        num_sweep_interfaces = self._effective_num_sweep(num_sweep_interfaces)

        # A and B: each is (f, sweep, 2 ports, virtual_cells, modes_out, modes_in)
        # Each field has 2 ports, so: 2 ports * virtual_cells * modes * modes
        if "A" in self.fields:
            bytes_total += (
                2
                * BYTES_COMPLEX
                * num_freqs
                * num_sweep_full
                * num_virtual_eme_cells
                * num_modes
                * num_modes
            )
        if "B" in self.fields:
            bytes_total += (
                2
                * BYTES_COMPLEX
                * num_freqs
                * num_sweep_full
                * num_virtual_eme_cells
                * num_modes
                * num_modes
            )

        # n_complex and flux: (f, sweep, cells, modes)
        if "n_complex" in self.fields:
            bytes_total += BYTES_COMPLEX * num_freqs * num_sweep_modes * num_eme_cells * num_modes
        if "flux" in self.fields:
            bytes_total += BYTES_COMPLEX * num_freqs * num_sweep_modes * num_eme_cells * num_modes

        # interface_smatrices: 4 S matrices (S11, S12, S21, S22), each (f, sweep, cells-1, modes, modes)
        if "interface_smatrices" in self.fields:
            bytes_total += (
                4
                * BYTES_COMPLEX
                * num_freqs
                * num_sweep_interfaces
                * num_interfaces
                * num_modes
                * num_modes
            )

        # overlaps: O11 (f, sweep, cells, modes, modes) + O12, O21 (f, sweep, cells-1, modes, modes)
        if "overlaps" in self.fields:
            bytes_total += (
                BYTES_COMPLEX * num_freqs * num_sweep_modes * num_eme_cells * num_modes * num_modes
            )  # O11
            bytes_total += (
                2
                * BYTES_COMPLEX
                * num_freqs
                * num_sweep_modes
                * num_interfaces
                * num_modes
                * num_modes
            )  # O12, O21

        return bytes_total


EMEMonitorType = (
    EMEModeSolverMonitor
    | EMEFieldMonitor
    | EMECoefficientMonitor
    | ModeSolverMonitor
    | PermittivityMonitor
    | MediumMonitor
)

"""Defines sweep settings for the EME simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import Field, PositiveInt, field_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.types import ArrayFloat1D, ArrayInt1D, ArrayLike
from tidy3d.exceptions import SetupError

from .grid import MAX_NUM_REPS


class EMESweepSpec(Tidy3dBaseModel, ABC):
    """Abstract spec for a parameter sweep during the EME propagation step.

    Notes
    -----
        An EME sweep re-runs the propagation step with varied parameters while reusing
        previously computed mode data. This makes sweeps much faster than running
        multiple independent simulations, since the mode solving step is typically the
        most expensive part of an EME simulation.

    See Also
    --------
        :class:`.EMELengthSweep` :
            Sweep over cell lengths.
        :class:`.EMEModeSweep` :
            Sweep over number of modes (convergence testing).
        :class:`.EMEPeriodicitySweep` :
            Sweep over number of periodic repetitions.
        :class:`.EMEFreqSweep` :
            Deprecated frequency sweep. Prefer ``EMESimulation.freqs`` with
            ``EMEModeSpec.interp_spec``.
    """

    @property
    @abstractmethod
    def num_sweep(self) -> PositiveInt:
        """Number of sweep indices."""

    @property
    def sweep_modes(self) -> bool:
        """Whether the sweep changes the modes."""
        return False

    @property
    def sweep_interfaces(self) -> bool:
        """Whether the sweep changes the cell interface scattering matrices."""
        return False

    @property
    def sweep_cells(self) -> bool:
        """Whether the sweep changes the propagation within a cell."""
        return False


class EMELengthSweep(EMESweepSpec):
    """Spec for sweeping EME cell lengths.

    Notes
    -----
        Varies the lengths of EME cells without re-solving modes. This is useful for
        optimizing device length, since only the propagation phase accumulated within
        each cell changes. If a 2D array is provided for ``scale_factors``, different
        cells can be scaled independently at each sweep index.

        For bent EME cells, this reuse remains valid only if the local mode problem is
        unchanged. In particular, bent anisotropic cells are rejected when changing the
        cell lengths would change the absolute tensor orientation seen by a reused mode,
        and bent custom media are only supported when no global-frame remapping of custom
        data is required.

    Example
    -------
    >>> sweep_spec = EMELengthSweep(scale_factors=[0.5, 1.0, 1.5, 2.0])
    """

    scale_factors: ArrayLike = Field(
        title="Length Scale Factor",
        description="Length scale factors to be used in the EME propagation step. "
        "The EME propagation step is repeated after scaling every cell length by this amount. "
        "The results are stored in 'sim_data.smatrix'. If a 2D array is provided, the "
        "first index is the sweep index and the second index is the cell index, "
        "allowing a nonuniform cell scaling along the propagation axis.",
    )

    @property
    def num_sweep(self) -> PositiveInt:
        """Number of sweep indices."""
        return len(self.scale_factors)

    @property
    def sweep_cells(self) -> bool:
        """Whether the sweep changes the propagation within a cell."""
        return True


class EMEModeSweep(EMESweepSpec):
    """Spec for sweeping number of modes in EME propagation step.
    Used for convergence testing.

    Example
    -------
    >>> sweep_spec = EMEModeSweep(num_modes=[1, 2, 5, 10])
    """

    num_modes: ArrayInt1D = Field(
        title="Number of Modes",
        description="Max number of trial / propagated modes to use in the EME propagation "
        "step. The EME propagation step is repeated after restricting the trial basis to "
        "modes with mode_index below this value, while the full EMEModeSpec.num_modes basis "
        "remains available for interface testing. Mode filters are applied after this cap "
        "without filling from higher mode indices. This can be used for convergence testing; "
        "reliable results should be independent of the number of trial modes used. This "
        "value cannot exceed the maximum number of modes in any EME cell in the simulation.",
    )

    @property
    def num_sweep(self) -> PositiveInt:
        """Number of sweep indices."""
        return len(self.num_modes)

    @property
    def sweep_interfaces(self) -> bool:
        """Whether the sweep changes the cell interface scattering matrices."""
        return True

    @property
    def sweep_cells(self) -> bool:
        """Whether the sweep changes the propagation within a cell."""
        return True


class EMEFreqSweep(EMESweepSpec):
    """Deprecated spec for sweeping frequency in the EME propagation step.

    Prefer specifying the target frequencies directly in ``EMESimulation.freqs`` and
    controlling the performance/accuracy tradeoff with ``EMEModeSpec.interp_spec``.
    ``EMEFreqSweep`` is kept for backward compatibility and uses a perturbative mode
    solver relative to the simulation EME modes.

    Example
    -------
    >>> sweep_spec = EMEFreqSweep(freq_scale_factors=[0.9, 0.95, 1.0, 1.05, 1.1])
    """

    freq_scale_factors: ArrayFloat1D = Field(
        title="Frequency Scale Factors",
        description="Deprecated approximate alternative to listing frequencies directly in "
        "``EMESimulation.freqs``. Scale factors are applied to every simulation frequency, "
        "and the new modes are then computed approximately using the exact modes as a basis. "
        "If there are multiple ``EMESimulation.freqs``, the exact modes are computed at each "
        "of those frequencies and then scaled independently.",
    )

    @property
    def num_sweep(self) -> PositiveInt:
        """Number of sweep indices."""
        return len(self.freq_scale_factors)

    @property
    def sweep_modes(self) -> bool:
        """Whether the sweep changes the modes."""
        return True

    @property
    def sweep_interfaces(self) -> bool:
        """Whether the sweep changes the cell interface scattering matrices."""
        return True

    @property
    def sweep_cells(self) -> bool:
        """Whether the sweep changes the propagation within a cell."""
        return True


class EMEPeriodicitySweep(EMESweepSpec):
    """Spec for sweeping number of repetitions of EME subgrids.

    Notes
    -----
        Useful for simulating long periodic structures like Bragg gratings,
        as it allows the EME solver to reuse the modes and cell interface
        scattering matrices.

        Compared to setting ``num_reps`` directly in the ``eme_grid_spec``,
        this sweep spec allows varying the number of repetitions,
        effectively simulating multiple structures in a single EME simulation.

        For bent EME cells, this reuse remains valid only if the local mode problem is
        unchanged in each repeated copy. Bent anisotropic cells are therefore rejected
        when the reused mode would see a different tensor orientation in another copy,
        and bent custom media are only supported when no global-frame remapping of custom
        data is required.

    Example
    -------
    >>> n_list = [1, 50, 100]
    >>> sweep_spec = EMEPeriodicitySweep(num_reps=[{"unit_cell": n} for n in n_list])
    """

    num_reps: list[dict[str, PositiveInt]] = Field(
        title="Number of Repetitions",
        description="Number of periodic repetitions of named subgrids in this EME grid. "
        "At each sweep index, contains a dict mapping the name of a subgrid to the "
        "number of repetitions of that subgrid at that sweep index.",
    )

    @field_validator("num_reps")
    @classmethod
    def _validate_num_reps(cls, val: list[dict[str, PositiveInt]]) -> list[dict[str, PositiveInt]]:
        """Check num_reps is not too large."""
        for num_reps_dict in val:
            for value in num_reps_dict.values():
                if value > MAX_NUM_REPS:
                    raise SetupError(
                        f"'EMEGridSpec' has 'num_reps={value:.2e}'; "
                        f"the largest value allowed is '{MAX_NUM_REPS}'."
                    )
        return val

    @property
    def num_sweep(self) -> PositiveInt:
        """Number of sweep indices."""
        return len(self.num_reps)


EMESweepSpecType = EMELengthSweep | EMEModeSweep | EMEFreqSweep | EMEPeriodicitySweep

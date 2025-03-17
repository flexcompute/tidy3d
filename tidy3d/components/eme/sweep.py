"""Defines sweep settings for the EME simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Union

import pydantic.v1 as pd

from ...exceptions import SetupError
from ..base import Tidy3dBaseModel
from ..types import ArrayFloat1D, ArrayInt1D, ArrayLike
from .grid import MAX_NUM_REPS


class EMESweepSpec(Tidy3dBaseModel, ABC):
    """Abstract spec for sweep done during EME propagation step."""

    @property
    @abstractmethod
    def num_sweep(self) -> pd.PositiveInt:
        """Number of sweep indices."""


class EMELengthSweep(EMESweepSpec):
    """Spec for sweeping EME cell lengths."""

    scale_factors: ArrayLike = pd.Field(
        ...,
        title="Length Scale Factor",
        description="Length scale factors to be used in the EME propagation step. "
        "The EME propagation step is repeated after scaling every cell length by this amount. "
        "The results are stored in 'sim_data.smatrix'. If a 2D array is provided, the "
        "first index is the sweep index and the second index is the cell index, "
        "allowing a nonuniform cell scaling along the propagation axis.",
    )

    @property
    def num_sweep(self) -> pd.PositiveInt:
        """Number of sweep indices."""
        return len(self.scale_factors)


class EMEModeSweep(EMESweepSpec):
    """Spec for sweeping number of modes in EME propagation step.
    Used for convergence testing."""

    num_modes: ArrayInt1D = pd.Field(
        ...,
        title="Number of Modes",
        description="Max number of modes to use in the EME propagation step. "
        "The EME propagation step is repeated after dropping modes with mode_index "
        "exceeding this value. This can be used for convergence testing; reliable results "
        "should be independent of the number of modes used. This value cannot exceed "
        "the maximum number of modes in any EME cell in the simulation.",
    )

    @property
    def num_sweep(self) -> pd.PositiveInt:
        """Number of sweep indices."""
        return len(self.num_modes)


class EMEFreqSweep(EMESweepSpec):
    """Spec for sweeping frequency in EME propagation step.
    Unlike ``sim.freqs``, the frequency sweep is approximate, using a
    perturbative mode solver relative to the simulation EME modes.
    This can be a faster way to solve at a larger number of frequencies."""

    freq_scale_factors: ArrayFloat1D = pd.Field(
        ...,
        title="Frequency Scale Factors",
        description="Scale factors "
        "applied to every frequency in 'EMESimulation.freqs'. After applying the scale factors, "
        "the new modes are computed approximately using the exact modes as a basis. "
        "If there are multiple 'EMESimulation.freqs', the exact modes are computed at each "
        "of those frequencies, and then the scale factors are applied to each independently.",
    )

    @property
    def num_sweep(self) -> pd.PositiveInt:
        """Number of sweep indices."""
        return len(self.freq_scale_factors)


class EMEPeriodicitySweep(EMESweepSpec):
    """Spec for sweeping number of repetitions of EME subgrids.
    Useful for simulating long periodic structures like Bragg gratings,
    as it allows the EME solver to reuse the modes and cell interface
    scattering matrices.

    Compared to setting ``num_reps`` directly in the ``eme_grid_spec``,
    this sweep spec allows varying the number of repetitions,
    effectively simulating multiple structures in a single EME simulation.

    Example
    -------
    >>> n_list = [1, 50, 100]
    >>> sweep_spec = EMEPeriodicitySweep(num_reps=[{"unit_cell": n} for n in n_list])
    """

    num_reps: List[Dict[str, pd.PositiveInt]] = pd.Field(
        ...,
        title="Number of Repetitions",
        description="Number of periodic repetitions of named subgrids in this EME grid. "
        "At each sweep index, contains a dict mapping the name of a subgrid to the "
        "number of repetitions of that subgrid at that sweep index.",
    )

    @pd.validator("num_reps", always=True)
    def _validate_num_reps(cls, val):
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
    def num_sweep(self) -> pd.PositiveInt:
        """Number of sweep indices."""
        return len(self.num_reps)


EMESweepSpecType = Union[EMELengthSweep, EMEModeSweep, EMEFreqSweep, EMEPeriodicitySweep]

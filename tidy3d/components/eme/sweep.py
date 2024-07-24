"""Defines sweep settings for the EME simulation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import pydantic.v1 as pd

from ..base import Tidy3dBaseModel
from ..types import ArrayFloat1D, ArrayInt1D, ArrayLike


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


EMESweepSpecType = Union[EMELengthSweep, EMEModeSweep, EMEFreqSweep]

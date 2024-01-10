"""Defines sweep settings for the EME simulation."""
from __future__ import annotations

from typing import List, Union
from abc import ABC

import pydantic.v1 as pd

from ..base import Tidy3dBaseModel


class EMESweepSpec(Tidy3dBaseModel, ABC):
    """Abstract spec for sweep done during EME propagation step."""

    pass


class EMELengthSweep(EMESweepSpec):
    """Spec for sweeping EME cell lengths.
    Currently, only a global scale factor is supported."""

    scale_factors: List[pd.PositiveFloat] = pd.Field(
        ...,
        title="Length Scale Factor",
        description="Length scale factors to be used in the EME propagation step. "
        "The EME propagation step is repeated after scaling every cell length by this amount. "
        "The results are stored in 'sim_data.smatrix'.",
    )


class EMEModeSweep(EMESweepSpec):
    """Spec for sweeping number of modes in EME propagation step.
    Used for convergence testing."""

    num_modes: List[pd.PositiveInt] = pd.Field(
        ...,
        title="Number of Modes",
        description="Max number of modes to use in the EME propagation step. "
        "The EME propagation step is repeated after dropping modes with mode_index "
        "exceeding this value. This can be used for convergence testing; reliable results "
        "should be independent of the number of modes used. This value cannot exceed "
        "the maximum number of modes in any EME cell in the simulation.",
    )


EMESweepSpecType = Union[EMELengthSweep, EMEModeSweep]

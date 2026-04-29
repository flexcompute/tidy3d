"""Union type for all source time profiles."""

from __future__ import annotations

from tidy3d.components.microwave.time import (
    BasebandCustomSourceTime,
    BasebandGaussianPulse,
    BasebandRectangularPulse,
    BasebandStep,
)
from tidy3d.components.source.time import (
    BroadbandPulse,
    ContinuousWave,
    CustomSourceTime,
    GaussianPulse,
)

SourceTimeType = (
    GaussianPulse
    | ContinuousWave
    | CustomSourceTime
    | BroadbandPulse
    | BasebandStep
    | BasebandGaussianPulse
    | BasebandRectangularPulse
    | BasebandCustomSourceTime
)

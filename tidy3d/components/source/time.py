"""Compatibility shim for :mod:`tidy3d._common.components.source.time`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.source.time import (
    _ROOTS_TOL,
    DEFAULT_SIGMA,
    END_TIME_FACTOR_GAUSSIAN,
    OFFSET_FWIDTH_FMAX,
    WARN_SOURCE_AMPLITUDE,
    BroadbandPulse,
    ContinuousWave,
    CustomSourceTime,
    GaussianPulse,
    Pulse,
    SourceTime,
    SourceTimeType,
)

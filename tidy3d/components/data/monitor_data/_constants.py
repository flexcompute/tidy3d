"""Constants shared by monitor-data implementations."""

# how much to shift the adjoint field source for 0-D axes dimensions
from __future__ import annotations

SHIFT_VALUE_ADJ_FLD_SRC = 1e-5
AXIAL_RATIO_CAP = 1e5
# At this sampling rate, the computed area of a sphere is within ~1% of the true value.
MIN_ANGULAR_SAMPLES_SPHERE = 10
MODE_INTERP_EXTRAPOLATION_TOLERANCE = 1e-2

__all__ = [
    "AXIAL_RATIO_CAP",
    "MIN_ANGULAR_SAMPLES_SPHERE",
    "MODE_INTERP_EXTRAPOLATION_TOLERANCE",
    "SHIFT_VALUE_ADJ_FLD_SRC",
]

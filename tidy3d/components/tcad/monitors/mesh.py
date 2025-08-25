"""Objects that define how data is recorded from simulation."""

from __future__ import annotations

from math import isclose
from typing import Literal

import pydantic.v1 as pd

from tidy3d.components.tcad.monitors.abstract import HeatChargeMonitor


class VolumeMeshMonitor(HeatChargeMonitor):
    """Monitor recording the volume mesh. The monitor size must be either 2D or 3D. If a 2D monitor
    is used in a 3D simulation, the sliced volumetric mesh on the plane of the monitor will be
    stored as a ``TriangularGridDataset``."""

    unstructured: Literal[True] = pd.Field(
        True,
        title="Unstructured Grid",
        description="Return the original unstructured grid.",
    )

    @pd.validator("size", always=True)
    def _at_least_2d(cls, val):
        """Validate that the monitor has at least two non-zero dimensions."""
        if len([d for d in val if isclose(d, 0)]) > 1:
            raise ValueError("'VolumeMeshMonitor' must have at least two nonzero dimensions.")
        return val

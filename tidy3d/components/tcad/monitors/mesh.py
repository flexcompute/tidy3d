"""Objects that define how data is recorded from simulation."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from tidy3d.components.tcad.monitors.abstract import HeatChargeMonitor


class VolumeMeshMonitor(HeatChargeMonitor):
    """Monitor recording the volume mesh. The monitor size must be either 2D or 3D. If a 2D monitor
    is used in a 3D simulation, the sliced volumetric mesh on the plane of the monitor will be
    stored as a ``TriangularGridDataset``."""

    unstructured: Literal[True] = Field(
        default=True,
        title="Unstructured Grid",
        description="Return the original unstructured grid.",
    )

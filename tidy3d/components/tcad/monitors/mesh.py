"""Objects that define how data is recorded from simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator

from tidy3d.components.tcad.monitors.abstract import HeatChargeMonitor

if TYPE_CHECKING:
    from tidy3d.components.autograd import TracedSize


class VolumeMeshMonitor(HeatChargeMonitor):
    """Monitor recording the volume mesh. The monitor size must be either 2D or 3D. If a 2D monitor
    is used in a 3D simulation, the sliced volumetric mesh on the plane of the monitor will be
    stored as a ``TriangularGridDataset``."""

    unstructured: Literal[True] = Field(
        default=True,
        title="Unstructured Grid",
        description="Return the original unstructured grid.",
    )

    @field_validator("size")
    @classmethod
    def _at_least_2d(cls, val: TracedSize) -> TracedSize:
        """Validate that the monitor has at least two non-zero dimensions."""
        from flexcompute.core._migration.tcad.validation import (
            validate_volume_mesh_monitor_size,
        )

        if diagnostics := validate_volume_mesh_monitor_size(val):
            raise ValueError(diagnostics[0].message)
        return val

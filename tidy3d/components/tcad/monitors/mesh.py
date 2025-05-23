"""Objects that define how data is recorded from simulation."""

from typing import Literal

import pydantic.v1 as pd

from tidy3d.components.tcad.monitors.abstract import HeatChargeMonitor


class VolumeMeshMonitor(HeatChargeMonitor):
    """Monitor for the volume mesh."""

    unstructured: Literal[True] = pd.Field(
        True,
        title="Unstructured Grid",
        description="Return the original unstructured grid.",
    )

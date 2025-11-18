from __future__ import annotations

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.tcad.monitors.mesh import VolumeMeshMonitor
from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation, TCADAnalysisTypes


class VolumeMesher(Tidy3dBaseModel):
    """Specification for a standalone volume mesher."""

    simulation: HeatChargeSimulation = pd.Field(
        ...,
        title="Simulation",
        description="HeatCharge simulation instance for the mesh specification.",
    )

    monitors: tuple[VolumeMeshMonitor, ...] = pd.Field(
        (),
        title="Monitors",
        description="List of monitors to be used for the mesher.",
    )

    def _get_simulation_types(self) -> list[TCADAnalysisTypes]:
        return [TCADAnalysisTypes.MESH]

    def validate_pre_upload(self):
        """Validate the VolumeMesher before uploading to the cloud.
        Currently no validation but method is required when calling ``web.upload``.
        """
        return

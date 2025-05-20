import pydantic as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation


class VolumeMeshSpec(Tidy3dBaseModel):
    """Specification for a standalone volume mesher."""

    simulation: HeatChargeSimulation = pd.Field(
        ..., description="HeatCharge simulation instance for the mesh specification."
    )

"""Defines heat simulation data class"""
from __future__ import annotations

import pydantic as pd

from .simulation import HeatSimulation

from ..data.data_array import SpatialDataArray
from ..base import Tidy3dBaseModel
from ..simulation import Simulation

from ...constants import KELVIN


class HeatSimulationData(Tidy3dBaseModel):
    """Stores results of a heat simulation.

    Example
    -------
    """

    heat_simulation: HeatSimulation = pd.Field(
        title="Heat Simulation",
        description="``HeatSimulation`` object describing the problem setup.",
    )

    temperature_data: SpatialDataArray = pd.Field(
        title="Temperature Field",
        description="Temperature field obtained from heat simulation.",
        units=KELVIN,
    )

    def perturbed_mediums_scene(self) -> Simulation:
        """Apply heat data to the original Tidy3D simulation (replaces appropriate media with CustomMedia). """

        return self.heat_simulation.scene.perturbed_mediums_copy(temperature=self.temperature_data)


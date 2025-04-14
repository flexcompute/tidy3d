from __future__ import annotations

from typing import Union

from tidy3d.components.data.monitor_data import ModeSolverData
from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.eme.data.sim_data import EMESimulationData
from tidy3d.components.eme.simulation import EMESimulation
from tidy3d.components.microwave.data.monitor_data import MicrowaveModeSolverData
from tidy3d.components.mode.data.sim_data import ModeSimulationData
from tidy3d.components.mode.simulation import ModeSimulation
from tidy3d.components.simulation import Simulation
from tidy3d.components.tcad.data.sim_data import (
    HeatChargeSimulationData,
    HeatSimulationData,
    VolumeMesherData,
)
from tidy3d.components.tcad.mesher import VolumeMesher
from tidy3d.components.tcad.simulation.heat import HeatSimulation
from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation
from tidy3d.plugins.mode.mode_solver import ModeSolver

SimulationType = Union[
    Simulation,
    HeatChargeSimulation,
    HeatSimulation,
    EMESimulation,
    ModeSolver,
    ModeSimulation,
    VolumeMesher,
]
SimulationDataType = Union[
    SimulationData,
    HeatChargeSimulationData,
    HeatSimulationData,
    EMESimulationData,
    MicrowaveModeSolverData,
    ModeSolverData,
    ModeSimulationData,
    VolumeMesherData,
]

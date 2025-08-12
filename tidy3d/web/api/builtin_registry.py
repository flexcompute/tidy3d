"""Register built-in simulation and data types with the web API registry.

This module is imported for its side effects by ``tidy3d.web.api.tidy3d_stub``.
"""

from __future__ import annotations

from tidy3d.components.data.sim_data import SimulationData
from tidy3d.components.eme.data.sim_data import EMESimulationData
from tidy3d.components.eme.simulation import EMESimulation
from tidy3d.components.mode.data.sim_data import ModeSimulationData, ModeSolverData
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
from tidy3d.web.core.constants import MODE_DATA_HDF5_GZ, MODE_FILE_HDF5_GZ
from tidy3d.web.core.types import TaskType

from .registry import (
    register_data_loader,
    register_remote_files,
    register_sim_loader,
    register_simulation_type,
)

# Class → TaskType
register_simulation_type(Simulation, TaskType.FDTD.name)
register_simulation_type(ModeSolver, TaskType.MODE_SOLVER.name)
register_simulation_type(ModeSimulation, TaskType.MODE.name)
register_simulation_type(HeatSimulation, TaskType.HEAT.name)
register_simulation_type(HeatChargeSimulation, TaskType.HEAT_CHARGE.name)
register_simulation_type(EMESimulation, TaskType.EME.name)
register_simulation_type(VolumeMesher, TaskType.VOLUME_MESH.name)

# JSON type → loaders for simulations
register_sim_loader("Simulation", Simulation.from_file)
register_sim_loader("ModeSolver", ModeSolver.from_file)
register_sim_loader("ModeSimulation", ModeSimulation.from_file)
register_sim_loader("HeatSimulation", HeatSimulation.from_file)
register_sim_loader("HeatChargeSimulation", HeatChargeSimulation.from_file)
register_sim_loader("EMESimulation", EMESimulation.from_file)
register_sim_loader("VolumeMesher", VolumeMesher.from_file)

# JSON type → loaders for data
register_data_loader("SimulationData", SimulationData.from_file)
register_data_loader("ModeSolverData", ModeSolverData.from_file)
register_data_loader("ModeSimulationData", ModeSimulationData.from_file)
register_data_loader("HeatSimulationData", HeatSimulationData.from_file)
register_data_loader("HeatChargeSimulationData", HeatChargeSimulationData.from_file)
register_data_loader("EMESimulationData", EMESimulationData.from_file)
register_data_loader("VolumeMesherData", VolumeMesherData.from_file)

# TaskType → custom remote files (only MODE_SOLVER deviates from default)
register_remote_files(TaskType.MODE_SOLVER.name, MODE_FILE_HDF5_GZ, MODE_DATA_HDF5_GZ)

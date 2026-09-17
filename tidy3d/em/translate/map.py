"""Lazy maps for backend Tidy3D schema translation."""

from __future__ import annotations

from enum import Enum
from importlib import import_module
from typing import Any


class Tidy3DSolverTaskMap(str, Enum):
    """Backend-local copy of public Tidy3D task type strings."""

    FDTD = "FDTD"
    MODE_SOLVER = "MODE_SOLVER"
    HEAT = "HEAT"
    HEAT_CHARGE = "HEAT_CHARGE"
    EME = "EME"
    MODE = "MODE"
    VOLUME_MESH = "VOLUME_MESH"
    MODAL_CM = "MODAL_CM"
    ADAM_OPTIMIZER = "ADAM_OPTIMIZER"


class Tidy3DSolverDataMap(str, Enum):
    """Backend-local result data family names."""

    FDTD = "FDTD"
    MODE_SOLVER = "MODE_SOLVER"
    MICROWAVE_MODE_SOLVER = "MICROWAVE_MODE_SOLVER"
    MODE = "MODE"
    EME = "EME"
    HEAT = "HEAT"
    HEAT_CHARGE = "HEAT_CHARGE"
    VOLUME_MESH = "VOLUME_MESH"
    MODAL_CM = "MODAL_CM"
    ADAM_OPTIMIZER = "ADAM_OPTIMIZER"


TASK_PUBLIC_CLASSES: dict[Tidy3DSolverTaskMap, str] = {
    Tidy3DSolverTaskMap.FDTD: "tidy3d:Simulation",
    Tidy3DSolverTaskMap.MODE_SOLVER: "tidy3d.plugins.mode:ModeSolver",
    Tidy3DSolverTaskMap.HEAT: "tidy3d:HeatSimulation",
    Tidy3DSolverTaskMap.HEAT_CHARGE: "tidy3d:HeatChargeSimulation",
    Tidy3DSolverTaskMap.EME: "tidy3d:EMESimulation",
    Tidy3DSolverTaskMap.MODE: "tidy3d:ModeSimulation",
    Tidy3DSolverTaskMap.VOLUME_MESH: "tidy3d:VolumeMesher",
    Tidy3DSolverTaskMap.MODAL_CM: (
        "tidy3d.plugins.smatrix.component_modelers.modal:ModalComponentModeler"
    ),
    Tidy3DSolverTaskMap.ADAM_OPTIMIZER: "tidy3d.plugins.invdes.optimizer:AdamOptimizer",
}

TASK_SCHEMA_CLASSES: dict[Tidy3DSolverTaskMap, str] = {
    Tidy3DSolverTaskMap.FDTD: "flexcompute.core._migration.em.schema.tidy3d.components.simulation:Simulation",
    Tidy3DSolverTaskMap.MODE_SOLVER: (
        "flexcompute.core._migration.em.schema.tidy3d.components.mode.mode_solver:ModeSolver"
    ),
    Tidy3DSolverTaskMap.HEAT: (
        "flexcompute.core._migration.em.schema.tidy3d.components.tcad.simulation.heat:HeatSimulation"
    ),
    Tidy3DSolverTaskMap.HEAT_CHARGE: (
        "flexcompute.core._migration.em.schema.tidy3d.components.tcad.simulation.heat_charge:HeatChargeSimulation"
    ),
    Tidy3DSolverTaskMap.EME: "flexcompute.core._migration.em.schema.tidy3d.components.eme.simulation:EMESimulation",
    Tidy3DSolverTaskMap.MODE: "flexcompute.core._migration.em.schema.tidy3d.components.mode.simulation:ModeSimulation",
    Tidy3DSolverTaskMap.VOLUME_MESH: "flexcompute.core._migration.em.schema.tidy3d.components.tcad.mesher:VolumeMesher",
    Tidy3DSolverTaskMap.MODAL_CM: (
        "flexcompute.core._migration.em.schema.tidy3d.plugins.smatrix.component_modelers.modal:ModalComponentModeler"
    ),
    Tidy3DSolverTaskMap.ADAM_OPTIMIZER: (
        "flexcompute.core._migration.em.schema.tidy3d.plugins.invdes.optimizer:AdamOptimizer"
    ),
}

TASK_CONVERTERS: dict[Tidy3DSolverTaskMap, tuple[str, str]] = {
    Tidy3DSolverTaskMap.FDTD: (
        "tidy3d.em.translate.fdtd.task:from_task",
        "tidy3d.em.translate.fdtd.task:to_task",
    ),
    Tidy3DSolverTaskMap.MODE_SOLVER: (
        "tidy3d.em.translate.mode_solver.task:from_task",
        "tidy3d.em.translate.mode_solver.task:to_task",
    ),
    Tidy3DSolverTaskMap.HEAT: (
        "tidy3d.em.translate.heat.task:from_task",
        "tidy3d.em.translate.heat.task:to_task",
    ),
    Tidy3DSolverTaskMap.HEAT_CHARGE: (
        "tidy3d.em.translate.heat_charge.task:from_task",
        "tidy3d.em.translate.heat_charge.task:to_task",
    ),
    Tidy3DSolverTaskMap.EME: (
        "tidy3d.em.translate.eme.task:from_task",
        "tidy3d.em.translate.eme.task:to_task",
    ),
    Tidy3DSolverTaskMap.MODE: (
        "tidy3d.em.translate.mode.task:from_task",
        "tidy3d.em.translate.mode.task:to_task",
    ),
    Tidy3DSolverTaskMap.VOLUME_MESH: (
        "tidy3d.em.translate.volume_mesh.task:from_task",
        "tidy3d.em.translate.volume_mesh.task:to_task",
    ),
    Tidy3DSolverTaskMap.MODAL_CM: (
        "tidy3d.em.translate.modal_cm.task:from_task",
        "tidy3d.em.translate.modal_cm.task:to_task",
    ),
    Tidy3DSolverTaskMap.ADAM_OPTIMIZER: (
        "tidy3d.em.translate.adam_optimizer.task:from_task",
        "tidy3d.em.translate.adam_optimizer.task:to_task",
    ),
}

DATA_PUBLIC_CLASSES: dict[Tidy3DSolverDataMap, str] = {
    Tidy3DSolverDataMap.FDTD: "tidy3d.components.data.sim_data:SimulationData",
    Tidy3DSolverDataMap.MICROWAVE_MODE_SOLVER: (
        "tidy3d.components.microwave.data.monitor_data:MicrowaveModeSolverData"
    ),
    Tidy3DSolverDataMap.MODE_SOLVER: "tidy3d.components.data.monitor_data:ModeSolverData",
    Tidy3DSolverDataMap.HEAT: "tidy3d.components.tcad.data.sim_data:HeatSimulationData",
    Tidy3DSolverDataMap.HEAT_CHARGE: (
        "tidy3d.components.tcad.data.sim_data:HeatChargeSimulationData"
    ),
    Tidy3DSolverDataMap.EME: "tidy3d.components.eme.data.sim_data:EMESimulationData",
    Tidy3DSolverDataMap.MODE: "tidy3d.components.mode.data.sim_data:ModeSimulationData",
    Tidy3DSolverDataMap.VOLUME_MESH: "tidy3d.components.tcad.data.sim_data:VolumeMesherData",
    Tidy3DSolverDataMap.MODAL_CM: "tidy3d.plugins.smatrix.data.modal:ModalComponentModelerData",
    Tidy3DSolverDataMap.ADAM_OPTIMIZER: "tidy3d.plugins.invdes.result:InverseDesignResult",
}

DATA_SCHEMA_CLASSES: dict[Tidy3DSolverDataMap, str] = {
    Tidy3DSolverDataMap.FDTD: "flexcompute.core._migration.em.schema.tidy3d.components.data.sim_data:SimulationData",
    Tidy3DSolverDataMap.MICROWAVE_MODE_SOLVER: (
        "flexcompute.core._migration.em.schema.tidy3d.components.microwave.data.monitor_data:MicrowaveModeSolverData"
    ),
    Tidy3DSolverDataMap.MODE_SOLVER: (
        "flexcompute.core._migration.em.schema.tidy3d.components.data.monitor_data:ModeSolverData"
    ),
    Tidy3DSolverDataMap.HEAT: (
        "flexcompute.core._migration.em.schema.tidy3d.components.tcad.data.sim_data:HeatSimulationData"
    ),
    Tidy3DSolverDataMap.HEAT_CHARGE: (
        "flexcompute.core._migration.em.schema.tidy3d.components.tcad.data.sim_data:HeatChargeSimulationData"
    ),
    Tidy3DSolverDataMap.EME: (
        "flexcompute.core._migration.em.schema.tidy3d.components.eme.data.sim_data:EMESimulationData"
    ),
    Tidy3DSolverDataMap.MODE: (
        "flexcompute.core._migration.em.schema.tidy3d.components.mode.data.sim_data:ModeSimulationData"
    ),
    Tidy3DSolverDataMap.VOLUME_MESH: (
        "flexcompute.core._migration.em.schema.tidy3d.components.tcad.data.sim_data:VolumeMesherData"
    ),
    Tidy3DSolverDataMap.MODAL_CM: (
        "flexcompute.core._migration.em.schema.tidy3d.plugins.smatrix.data.modal:ModalComponentModelerData"
    ),
    Tidy3DSolverDataMap.ADAM_OPTIMIZER: "flexcompute.core._migration.em.schema.tidy3d.plugins.invdes.result:InverseDesignResult",
}

DATA_CONVERTERS: dict[Tidy3DSolverDataMap, tuple[str, str]] = {
    Tidy3DSolverDataMap.FDTD: (
        "tidy3d.em.translate.fdtd.data:from_data",
        "tidy3d.em.translate.fdtd.data:to_data",
    ),
    Tidy3DSolverDataMap.MODE_SOLVER: (
        "tidy3d.em.translate.mode_solver.data:from_data",
        "tidy3d.em.translate.mode_solver.data:to_data",
    ),
    Tidy3DSolverDataMap.MICROWAVE_MODE_SOLVER: (
        "tidy3d.em.translate.mode_solver.data:from_data",
        "tidy3d.em.translate.mode_solver.data:to_data",
    ),
    Tidy3DSolverDataMap.HEAT: (
        "tidy3d.em.translate.heat.data:from_data",
        "tidy3d.em.translate.heat.data:to_data",
    ),
    Tidy3DSolverDataMap.HEAT_CHARGE: (
        "tidy3d.em.translate.heat_charge.data:from_data",
        "tidy3d.em.translate.heat_charge.data:to_data",
    ),
    Tidy3DSolverDataMap.EME: (
        "tidy3d.em.translate.eme.data:from_data",
        "tidy3d.em.translate.eme.data:to_data",
    ),
    Tidy3DSolverDataMap.MODE: (
        "tidy3d.em.translate.mode.data:from_data",
        "tidy3d.em.translate.mode.data:to_data",
    ),
    Tidy3DSolverDataMap.VOLUME_MESH: (
        "tidy3d.em.translate.volume_mesh.data:from_data",
        "tidy3d.em.translate.volume_mesh.data:to_data",
    ),
    Tidy3DSolverDataMap.MODAL_CM: (
        "tidy3d.em.translate.modal_cm.data:from_data",
        "tidy3d.em.translate.modal_cm.data:to_data",
    ),
    Tidy3DSolverDataMap.ADAM_OPTIMIZER: (
        "tidy3d.em.translate.adam_optimizer.data:from_data",
        "tidy3d.em.translate.adam_optimizer.data:to_data",
    ),
}


def resolve(path: str) -> Any:
    """Resolve a ``module:attribute`` path lazily."""

    module_name, attr = path.split(":", 1)
    return getattr(import_module(module_name), attr)

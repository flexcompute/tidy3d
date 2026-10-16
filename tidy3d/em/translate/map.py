"""Lazy maps for backend Tidy3D schema translation."""

from __future__ import annotations

from enum import Enum
from importlib import import_module
from typing import Any, NamedTuple


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
        "flexcompute.core._migration.solver.component_modeler.schema:ComponentModelerTaskInput"
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
    Tidy3DSolverDataMap.ADAM_OPTIMIZER: (
        "tidy3d.em.translate.adam_optimizer.data:from_data",
        "tidy3d.em.translate.adam_optimizer.data:to_data",
    ),
}


class ComponentScope(NamedTuple):
    """Where a public class translates to, which facade checks it, and when.

    ``on_construction`` runs the full facade after every construction of the public
    class. Client policy validators stay on the public class.
    """

    schema: str
    facade: str
    on_construction: bool = False


# Component registration stays here, not on public model classes. Abstract roots
# mark families whose new subclasses must register their own schema explicitly.
COMPONENT_SCOPES: dict[str, ComponentScope | None] = {
    "tidy3d.components.eme.grid:EMEGridSpec": None,
    "tidy3d.components.eme.sweep:EMESweepSpec": None,
    "tidy3d.components.eme.grid:EMEModeSpec": ComponentScope(
        "flexcompute.core._migration.em.eme.schema.grid:EMEModeSpec",
        "flexcompute.core._migration.em.eme.validation:validate",
    ),
    "tidy3d.components.eme.grid:EMEUniformGrid": ComponentScope(
        "flexcompute.core._migration.em.eme.schema.grid:EMEUniformGrid",
        "flexcompute.core._migration.em.eme.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.eme.grid:EMEExplicitGrid": ComponentScope(
        "flexcompute.core._migration.em.eme.schema.grid:EMEExplicitGrid",
        "flexcompute.core._migration.em.eme.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.eme.grid:EMECompositeGrid": ComponentScope(
        "flexcompute.core._migration.em.eme.schema.grid:EMECompositeGrid",
        "flexcompute.core._migration.em.eme.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.eme.grid:EMEGrid": ComponentScope(
        "flexcompute.core._migration.em.eme.schema.grid:EMEGrid",
        "flexcompute.core._migration.em.eme.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.eme.sweep:EMELengthSweep": ComponentScope(
        "flexcompute.core._migration.em.eme.schema.sweep:EMELengthSweep",
        "flexcompute.core._migration.em.eme.validation:validate",
    ),
    "tidy3d.components.eme.sweep:EMEModeSweep": ComponentScope(
        "flexcompute.core._migration.em.eme.schema.sweep:EMEModeSweep",
        "flexcompute.core._migration.em.eme.validation:validate",
    ),
    "tidy3d.components.eme.sweep:EMEPeriodicitySweep": ComponentScope(
        "flexcompute.core._migration.em.eme.schema.sweep:EMEPeriodicitySweep",
        "flexcompute.core._migration.em.eme.validation:validate",
        on_construction=True,
    ),
    # medium
    "tidy3d.components.nonlinear:NonlinearSpec": ComponentScope(
        "flexcompute.core._migration.em.material.nonlinear.schema.nonlinear:NonlinearSpec",
        "flexcompute.core._migration.em.material.nonlinear.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.nonlinear:NonlinearSusceptibility": ComponentScope(
        "flexcompute.core._migration.em.material.nonlinear.schema.nonlinear:NonlinearSusceptibility",
        "flexcompute.core._migration.em.material.nonlinear.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.time_modulation:SpaceModulation": ComponentScope(
        "flexcompute.core._migration.em.material.time_modulation.schema.time_modulation:SpaceModulation",
        "flexcompute.core._migration.em.material.time_modulation.validation:validate",
    ),
    "tidy3d.components.time_modulation:ModulationSpec": ComponentScope(
        "flexcompute.core._migration.em.material.time_modulation.schema.time_modulation:ModulationSpec",
        "flexcompute.core._migration.em.material.time_modulation.validation:validate",
        on_construction=True,
    ),
    # tcad
    "tidy3d.components.tcad.grid:_GridRefinementLineBase": None,
    "tidy3d.components.tcad.grid:_GridRefinementRegionBase": None,
    "tidy3d.components.tcad.mobility:_HighFieldDependenceBase": None,
    "tidy3d.components.tcad.mobility:CanaliFieldDependence": ComponentScope(
        "flexcompute.core._migration.tcad.schema.mobility:CanaliFieldDependence",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.mobility:TransferredElectronFieldDependence": ComponentScope(
        "flexcompute.core._migration.tcad.schema.mobility:TransferredElectronFieldDependence",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.mobility:LombardiSurfaceDependence": ComponentScope(
        "flexcompute.core._migration.tcad.schema.mobility:LombardiSurfaceDependence",
        "flexcompute.core._migration.tcad.validation:validate",
    ),
    "tidy3d.components.tcad.mobility:MasettiMobility": ComponentScope(
        "flexcompute.core._migration.tcad.schema.mobility:MasettiMobility",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.mobility:CaugheyThomasMobility": ComponentScope(
        "flexcompute.core._migration.tcad.schema.mobility:CaugheyThomasMobility",
        "flexcompute.core._migration.tcad.validation:validate",
    ),
    "tidy3d.components.tcad.mobility:AlloyMobility": ComponentScope(
        "flexcompute.core._migration.tcad.schema.mobility:AlloyMobility",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.generation_recombination:FossumCarrierLifetime": ComponentScope(
        "flexcompute.core._migration.tcad.schema.generation_recombination:FossumCarrierLifetime",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.generation_recombination:HurkxTrapAssistedTunneling": ComponentScope(
        "flexcompute.core._migration.tcad.schema.generation_recombination:HurkxTrapAssistedTunneling",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.generation_recombination:SurfaceShockleyReedHallRecombination": ComponentScope(
        "flexcompute.core._migration.tcad.schema.generation_recombination:SurfaceShockleyReedHallRecombination",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.generation_recombination:HeterojunctionCaptureSide": ComponentScope(
        "flexcompute.core._migration.tcad.schema.generation_recombination:HeterojunctionCaptureSide",
        "flexcompute.core._migration.tcad.validation:validate",
    ),
    "tidy3d.components.tcad.generation_recombination:HeterojunctionSurfaceShockleyReedHallRecombination": ComponentScope(
        "flexcompute.core._migration.tcad.schema.generation_recombination:HeterojunctionSurfaceShockleyReedHallRecombination",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.generation_recombination:DistributedGeneration": ComponentScope(
        "flexcompute.core._migration.tcad.schema.generation_recombination:DistributedGeneration",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.grid:GridRefinementLine": ComponentScope(
        "flexcompute.core._migration.tcad.schema.grid:GridRefinementLine",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.grid:RelativeGridRefinementLine": ComponentScope(
        "flexcompute.core._migration.tcad.schema.grid:RelativeGridRefinementLine",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.grid:GridRefinementRegion": ComponentScope(
        "flexcompute.core._migration.tcad.schema.grid:GridRefinementRegion",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.grid:RelativeGridRefinementRegion": ComponentScope(
        "flexcompute.core._migration.tcad.schema.grid:RelativeGridRefinementRegion",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.grid:DistanceUnstructuredGrid": ComponentScope(
        "flexcompute.core._migration.tcad.schema.grid:DistanceUnstructuredGrid",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.grid:AutoUnstructuredGrid": ComponentScope(
        "flexcompute.core._migration.tcad.schema.grid:AutoUnstructuredGrid",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    # invdes: the family translates, but only the random spec validates on construction. The custom
    # spec keeps client validators (see there) and the uniform spec has none.
    "tidy3d.plugins.invdes.initialization:AbstractInitializationSpec": None,
    "tidy3d.plugins.invdes.initialization:RandomInitializationSpec": ComponentScope(
        "flexcompute.core._migration.em.invdes.schema.initialization:RandomInitializationSpec",
        "flexcompute.core._migration.em.invdes.validation:validate_initialization",
        on_construction=True,
    ),
    "tidy3d.plugins.invdes.initialization:UniformInitializationSpec": ComponentScope(
        "flexcompute.core._migration.em.invdes.schema.initialization:UniformInitializationSpec",
        "flexcompute.core._migration.em.invdes.validation:validate_initialization",
    ),
    "tidy3d.plugins.invdes.initialization:CustomInitializationSpec": ComponentScope(
        "flexcompute.core._migration.em.invdes.schema.initialization:CustomInitializationSpec",
        "flexcompute.core._migration.em.invdes.validation:validate_initialization",
    ),
}

COMPONENT_ERROR_TYPES = {
    "em.eme.grid_repetitions_exceeded": "tidy3d.exceptions:SetupError",
    "em.eme.periodicity_repetitions_exceeded": "tidy3d.exceptions:SetupError",
    "em.eme.constructed_grid_cells_exceeded": "tidy3d.exceptions:SetupError",
    "em.eme.constructed_grid_modes_exceeded": "tidy3d.exceptions:SetupError",
    "em.eme.constructed_grid_boundary_count": "tidy3d.exceptions:ValidationError",
    "em.eme.constructed_grid_first_boundary_outside": "tidy3d.exceptions:ValidationError",
    "em.eme.constructed_grid_boundary_order": "tidy3d.exceptions:ValidationError",
    "em.eme.constructed_grid_last_boundary_outside": "tidy3d.exceptions:ValidationError",
    # medium
    "em.material.nonlinear.unique_models": "tidy3d.exceptions:ValidationError",
    "em.material.nonlinear.consistent_n0": "tidy3d.exceptions:SetupError",
    "em.material.nonlinear.max_num_iters": "tidy3d.exceptions:ValidationError",
    "em.material.nonlinear.max_numiters": "tidy3d.exceptions:ValidationError",
    # tcad
    "tcad.refinement_line_infinite_point": "tidy3d.exceptions:ValidationError",
    # invdes
    "em.invdes.initialization_bound_order": "tidy3d.exceptions:ValidationError",
}


def resolve(path: str) -> Any:
    """Resolve a ``module:attribute`` path lazily."""

    module_name, attr = path.split(":", 1)
    return getattr(import_module(module_name), attr)

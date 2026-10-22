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
    # ``import tidy3d`` builds these, so they stay off the construction path.
    "tidy3d.components.medium.isotropic:PECMedium": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.isotropic:PECMedium",
        "flexcompute.core._migration.em.material.medium.validation:validate",
    ),
    "tidy3d.components.medium.isotropic:PMCMedium": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.isotropic:PMCMedium",
        "flexcompute.core._migration.em.material.medium.validation:validate",
    ),
    # Off the construction path: field order (see ``_validate_frequency_range``).
    "tidy3d.components.medium.lossy_metal:LossyMetalMedium": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.lossy_metal:LossyMetalMedium",
        "flexcompute.core._migration.em.material.medium.validation:validate",
    ),
    # Registered because the plan registers every moved class. The roughness models carry no
    # rules, so they stay off the construction path.
    "tidy3d.components.medium.roughness:AbstractSurfaceRoughness": None,
    "tidy3d.components.medium.roughness:HammerstadSurfaceRoughness": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.roughness:HammerstadSurfaceRoughness",
        "flexcompute.core._migration.em.material.medium.validation:validate",
    ),
    "tidy3d.components.medium.roughness:HuraySurfaceRoughness": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.roughness:HuraySurfaceRoughness",
        "flexcompute.core._migration.em.material.medium.validation:validate",
    ),
    "tidy3d.components.medium.roughness:SurfaceImpedanceFitterParam": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.roughness:SurfaceImpedanceFitterParam",
        "flexcompute.core._migration.em.material.medium.validation:validate",
    ),
    "tidy3d.components.medium.anisotropic:FullyAnisotropicMedium": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.anisotropic:FullyAnisotropicMedium",
        "flexcompute.core._migration.em.material.medium.validation:validate",
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
    "tidy3d.components.tcad.boundary.heat:ThermalContactResistance": ComponentScope(
        "flexcompute.core._migration.tcad.schema.boundary:ThermalContactResistance",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.monitors.mesh:VolumeMeshMonitor": ComponentScope(
        "flexcompute.core._migration.tcad.schema.monitor:VolumeMeshMonitor",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.source.abstract:StructureBasedHeatChargeSource": None,
    "tidy3d.components.tcad.source.heat:HeatSource": ComponentScope(
        "flexcompute.core._migration.tcad.schema.source:HeatSource",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.source.heat:UniformHeatSource": ComponentScope(
        "flexcompute.core._migration.tcad.schema.source:UniformHeatSource",
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
    "tidy3d.plugins.invdes.region:DesignRegion": None,
    "tidy3d.plugins.invdes.region:TopologyDesignRegion": ComponentScope(
        "flexcompute.core._migration.em.invdes.schema.region:TopologyDesignRegion",
        "flexcompute.core._migration.em.invdes.validation:validate_region",
    ),
    "tidy3d.plugins.invdes.design:AbstractInverseDesign": None,
    "tidy3d.plugins.invdes.design:InverseDesign": ComponentScope(
        "flexcompute.core._migration.em.invdes.schema.design:InverseDesign",
        "flexcompute.core._migration.em.invdes.validation:validate_design",
    ),
    "tidy3d.plugins.invdes.design:InverseDesignMulti": ComponentScope(
        "flexcompute.core._migration.em.invdes.schema.design:InverseDesignMulti",
        "flexcompute.core._migration.em.invdes.validation:validate_design",
        on_construction=True,
    ),
    # monitor
    "tidy3d.components.apodization:ApodizationSpec": ComponentScope(
        "flexcompute.core._migration.em.apodization.schema:ApodizationSpec",
        "flexcompute.core._migration.em.apodization.validation:validate",
    ),
    "tidy3d.components.monitor:SurfaceIntegrationMonitor": None,
    "tidy3d.components.monitor:AbstractFluxMonitor": None,
    "tidy3d.components.monitor:AbstractFieldProjectionMonitor": None,
    "tidy3d.components.monitor:AbstractSurfaceMonitor": None,
    "tidy3d.components.monitor:TimeMonitor": None,
    "tidy3d.components.monitor:FluxMonitor": ComponentScope(
        "flexcompute.core._migration.em.monitor.schema:FluxMonitor",
        "flexcompute.core._migration.em.monitor.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.monitor:FluxTimeMonitor": ComponentScope(
        "flexcompute.core._migration.em.monitor.schema:FluxTimeMonitor",
        "flexcompute.core._migration.em.monitor.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.monitor:FieldTimeMonitor": ComponentScope(
        "flexcompute.core._migration.em.monitor.schema:FieldTimeMonitor",
        "flexcompute.core._migration.em.monitor.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.monitor:AuxFieldTimeMonitor": ComponentScope(
        "flexcompute.core._migration.em.monitor.schema:AuxFieldTimeMonitor",
        "flexcompute.core._migration.em.monitor.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.monitor:ModeTimeMonitor": ComponentScope(
        "flexcompute.core._migration.em.monitor.schema:ModeTimeMonitor",
        "flexcompute.core._migration.em.monitor.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.material.tcad.heat:AbstractThermalPropertyModel": None,
    "tidy3d.components.material.tcad.heat:PowerLawConductivity": ComponentScope(
        "flexcompute.core._migration.tcad.schema.heat_medium:PowerLawConductivity",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.material.tcad.heat:PolynomialConductivity": ComponentScope(
        "flexcompute.core._migration.tcad.schema.heat_medium:PolynomialConductivity",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.material.tcad.heat:InversePolynomialConductivity": ComponentScope(
        "flexcompute.core._migration.tcad.schema.heat_medium:InversePolynomialConductivity",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.material.tcad.heat:PowerLawHeatCapacity": ComponentScope(
        "flexcompute.core._migration.tcad.schema.heat_medium:PowerLawHeatCapacity",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.material.tcad.heat:PolynomialHeatCapacity": ComponentScope(
        "flexcompute.core._migration.tcad.schema.heat_medium:PolynomialHeatCapacity",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.material.tcad.heat:SolidMedium": ComponentScope(
        "flexcompute.core._migration.tcad.schema.heat_medium:SolidMedium",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.material.tcad.heat:SolidSpec": ComponentScope(
        "flexcompute.core._migration.tcad.schema.heat_medium:SolidSpec",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.doping:AbstractDopingBox": None,
    "tidy3d.components.tcad.doping:ConstantDoping": ComponentScope(
        "flexcompute.core._migration.tcad.schema.doping:ConstantDoping",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.doping:GaussianDoping": ComponentScope(
        "flexcompute.core._migration.tcad.schema.doping:GaussianDoping",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.doping:CustomDoping": ComponentScope(
        "flexcompute.core._migration.tcad.schema.doping:CustomDoping",
        "flexcompute.core._migration.tcad.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.monitor:FieldProjectionAngleMonitor": ComponentScope(
        "flexcompute.core._migration.em.monitor.schema:FieldProjectionAngleMonitor",
        "flexcompute.core._migration.em.monitor.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.monitor:FieldProjectionCartesianMonitor": ComponentScope(
        "flexcompute.core._migration.em.monitor.schema:FieldProjectionCartesianMonitor",
        "flexcompute.core._migration.em.monitor.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.monitor:FieldProjectionKSpaceMonitor": ComponentScope(
        "flexcompute.core._migration.em.monitor.schema:FieldProjectionKSpaceMonitor",
        "flexcompute.core._migration.em.monitor.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.monitor:DirectivityMonitor": ComponentScope(
        "flexcompute.core._migration.em.monitor.schema:DirectivityMonitor",
        "flexcompute.core._migration.em.monitor.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.monitor:DiffractionMonitor": ComponentScope(
        "flexcompute.core._migration.em.monitor.schema:DiffractionMonitor",
        "flexcompute.core._migration.em.monitor.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.monitor:FieldProjectionSurface": ComponentScope(
        "flexcompute.core._migration.em.monitor.schema:FieldProjectionSurface",
        "flexcompute.core._migration.em.monitor.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.monitor:SurfaceFieldMonitor": ComponentScope(
        "flexcompute.core._migration.em.monitor.schema:SurfaceFieldMonitor",
        "flexcompute.core._migration.em.monitor.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.monitor:SurfaceFieldStructureMonitor": ComponentScope(
        "flexcompute.core._migration.em.monitor.schema:SurfaceFieldStructureMonitor",
        "flexcompute.core._migration.em.monitor.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.monitor:SurfaceFieldTimeMonitor": ComponentScope(
        "flexcompute.core._migration.em.monitor.schema:SurfaceFieldTimeMonitor",
        "flexcompute.core._migration.em.monitor.validation:validate",
        on_construction=True,
    ),
    # grid: layer refinement
    "tidy3d.components.grid.grid_spec.refinement.model:LayerRefinementSpec": ComponentScope(
        "flexcompute.core._migration.em.grid.schema.grid_spec.refinement.model:LayerRefinementSpec",
        "flexcompute.core._migration.em.grid.validation:validate",
        on_construction=True,
    ),
    # rf: lumped elements
    "tidy3d.components.lumped_element:LumpedElement": None,
    "tidy3d.components.lumped_element:_AdmittanceFitter": None,
    "tidy3d.components.lumped_element:LumpedResistor": ComponentScope(
        "flexcompute.core._migration.em.lumped_element.schema:LumpedResistor",
        "flexcompute.core._migration.em.lumped_element.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.lumped_element:CoaxialLumpedResistor": ComponentScope(
        "flexcompute.core._migration.em.lumped_element.schema:CoaxialLumpedResistor",
        "flexcompute.core._migration.em.lumped_element.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.lumped_element:LinearLumpedElement": ComponentScope(
        "flexcompute.core._migration.em.lumped_element.schema:LinearLumpedElement",
        "flexcompute.core._migration.em.lumped_element.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.lumped_element:RLCNetwork": ComponentScope(
        "flexcompute.core._migration.em.lumped_element.schema:RLCNetwork",
        "flexcompute.core._migration.em.lumped_element.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.lumped_element:AdmittanceNetwork": ComponentScope(
        "flexcompute.core._migration.em.lumped_element.schema:AdmittanceNetwork",
        "flexcompute.core._migration.em.lumped_element.validation:validate",
    ),
    "tidy3d.components.lumped_element:CircuitImpedanceModel": ComponentScope(
        "flexcompute.core._migration.em.lumped_element.schema:CircuitImpedanceModel",
        "flexcompute.core._migration.em.lumped_element.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.lumped_element:LumpedCircuitComponent": ComponentScope(
        "flexcompute.core._migration.em.lumped_element.schema:LumpedCircuitComponent",
        "flexcompute.core._migration.em.lumped_element.validation:validate",
        on_construction=True,
    ),
    # frequency-extrapolation; FlexRF's subclass translates through the abstract class
    "tidy3d.components.frequency_extrapolation:AbstractLowFrequencySmoothingSpec": ComponentScope(
        "flexcompute.core._migration.em.frequency_extrapolation.schema:AbstractLowFrequencySmoothingSpec",
        "flexcompute.core._migration.em.frequency_extrapolation.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.frequency_extrapolation:LowFrequencySmoothingSpec": ComponentScope(
        "flexcompute.core._migration.em.frequency_extrapolation.schema:LowFrequencySmoothingSpec",
        "flexcompute.core._migration.em.frequency_extrapolation.validation:validate",
        on_construction=True,
    ),
    # scene: transformation
    "tidy3d.components.transformation:AbstractRotation": None,
    "tidy3d.components.transformation:AbstractReflection": None,
    "tidy3d.components.transformation:RotationAroundAxis": ComponentScope(
        "flexcompute.core._migration.em.transformation.schema:RotationAroundAxis",
        "flexcompute.core._migration.em.transformation.validation:validate",
    ),
    "tidy3d.components.transformation:ReflectionFromPlane": ComponentScope(
        "flexcompute.core._migration.em.transformation.schema:ReflectionFromPlane",
        "flexcompute.core._migration.em.transformation.validation:validate",
    ),
    # rf: path integrals and impedance
    "tidy3d.components.microwave.path_integrals.specs.base:AbstractAxesRH": None,
    "tidy3d.components.microwave.path_integrals.specs.impedance:AbstractImpedanceSpec": None,
    "tidy3d.components.microwave.path_integrals.mode_plane_analyzer:ModePlaneAnalyzer": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.mode_plane_analyzer:ModePlaneAnalyzer",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.path_integrals.specs.base:AxisAlignedPathIntegralSpec": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.specs.base:AxisAlignedPathIntegralSpec",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.path_integrals.specs.base:Custom2DPathIntegralSpec": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.specs.base:Custom2DPathIntegralSpec",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.path_integrals.specs.voltage:AxisAlignedVoltageIntegralSpec": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.specs.voltage:AxisAlignedVoltageIntegralSpec",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.path_integrals.specs.voltage:Custom2DVoltageIntegralSpec": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.specs.voltage:Custom2DVoltageIntegralSpec",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.path_integrals.specs.current:AxisAlignedCurrentIntegralSpec": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.specs.current:AxisAlignedCurrentIntegralSpec",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.path_integrals.specs.current:Custom2DCurrentIntegralSpec": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.specs.current:Custom2DCurrentIntegralSpec",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.path_integrals.specs.current:CompositeCurrentIntegralSpec": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.specs.current:CompositeCurrentIntegralSpec",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.path_integrals.specs.impedance:AutoImpedanceSpec": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.specs.impedance:AutoImpedanceSpec",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.path_integrals.specs.impedance:CustomImpedanceSpec": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.specs.impedance:CustomImpedanceSpec",
        "flexcompute.core._migration.em.microwave.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.microwave.path_integrals.integrals.base:AxisAlignedPathIntegral": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.integrals.base:AxisAlignedPathIntegral",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.path_integrals.integrals.base:Custom2DPathIntegral": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.integrals.base:Custom2DPathIntegral",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.path_integrals.integrals.voltage:AxisAlignedVoltageIntegral": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.integrals.voltage:AxisAlignedVoltageIntegral",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.path_integrals.integrals.voltage:Custom2DVoltageIntegral": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.integrals.voltage:Custom2DVoltageIntegral",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.path_integrals.integrals.current:AxisAlignedCurrentIntegral": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.integrals.current:AxisAlignedCurrentIntegral",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.path_integrals.integrals.current:Custom2DCurrentIntegral": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.integrals.current:Custom2DCurrentIntegral",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.path_integrals.integrals.current:CompositeCurrentIntegral": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.path_integrals.integrals.current:CompositeCurrentIntegral",
        "flexcompute.core._migration.em.microwave.validation:validate",
    ),
    "tidy3d.components.microwave.impedance_calculator:ImpedanceCalculator": ComponentScope(
        "flexcompute.core._migration.em.microwave.schema.impedance_calculator:ImpedanceCalculator",
        "flexcompute.core._migration.em.microwave.validation:validate",
        on_construction=True,
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
    "em.material.medium.anisotropic_component_nonlinear": "tidy3d.exceptions:ValidationError",
    "em.material.medium.two_d_component_nonlinear": "tidy3d.exceptions:ValidationError",
    "em.material.medium.nonlinear_numiters_deprecated": "tidy3d.exceptions:ValidationError",
    "em.material.medium.modulation_and_nonlinear": "tidy3d.exceptions:ValidationError",
    "em.material.medium.two_d_component_modulation": "tidy3d.exceptions:ValidationError",
    "em.material.nonlinear.medium_time_modulated": "tidy3d.exceptions:ValidationError",
    "em.material.nonlinear.medium_anisotropic": "tidy3d.exceptions:ValidationError",
    "em.material.nonlinear.medium_class": "tidy3d.exceptions:ValidationError",
    "em.material.nonlinear.two_photon_absorption_passive": "tidy3d.exceptions:ValidationError",
    "em.material.medium.anisotropic_permittivity_symmetric": "tidy3d.exceptions:ValidationError",
    "em.material.medium.anisotropic_permittivity_ge_one": "tidy3d.exceptions:ValidationError",
    "em.material.medium.modulation_spec_unsupported": "tidy3d.exceptions:ValidationError",
    # tcad
    "tcad.refinement_line_infinite_point": "tidy3d.exceptions:ValidationError",
    # invdes
    "em.invdes.initialization_bound_order": "tidy3d.exceptions:ValidationError",
    "em.invdes.multi_mirror_symmetry": "tidy3d.exceptions:ValidationError",
    # monitor
    "em.monitor.window_size_for_surface": "tidy3d.exceptions:ValidationError",
    "em.monitor.exclude_surfaces_on_surface": "tidy3d.exceptions:SetupError",
    "em.monitor.stop_before_start": "tidy3d.exceptions:SetupError",
    "tcad.doping_box_requires_3d": "tidy3d.exceptions:SetupError",
    "tcad.nonfinite_thermal_resistance": "builtins:ValueError",
    "tcad.volume_mesh_monitor_at_least_2d": "builtins:ValueError",
    "tcad.heat_charge_source_requires_structure": "tidy3d.exceptions:SetupError",
    # frequency-extrapolation
    "em.frequency_extrapolation.empty_monitors": "builtins:ValueError",
    # rf: path integrals and impedance
    "em.microwave.missing_integrals": "tidy3d.exceptions:ValidationError",
}


def resolve(path: str) -> Any:
    """Resolve a ``module:attribute`` path lazily."""

    module_name, attr = path.split(":", 1)
    return getattr(import_module(module_name), attr)

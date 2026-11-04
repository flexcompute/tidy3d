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
    Tidy3DSolverTaskMap.FDTD: "flexcompute.core._migration.solver.fdtd.schema.model:Simulation",
    Tidy3DSolverTaskMap.MODE_SOLVER: (
        "flexcompute.core._migration.solver.mode.schema.mode_solver:ModeSolver"
    ),
    Tidy3DSolverTaskMap.HEAT: (
        "flexcompute.core._migration.solver.heat_charge.schema.heat:HeatSimulation"
    ),
    Tidy3DSolverTaskMap.HEAT_CHARGE: (
        "flexcompute.core._migration.solver.heat_charge.schema.heat_charge:HeatChargeSimulation"
    ),
    Tidy3DSolverTaskMap.EME: "flexcompute.core._migration.solver.eme.schema.simulation:EMESimulation",
    Tidy3DSolverTaskMap.MODE: "flexcompute.core._migration.solver.mode.schema.simulation:ModeSimulation",
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
    class. Client policy validators stay on the public class. ``translator`` names a step
    the translation applies to the class's dumped data, wherever it is nested, before the
    schema validates it, such as a fold, so the rules check the translated form.
    """

    schema: str
    facade: str
    on_construction: bool = False
    translator: str | None = None


# The FDTD, EME and mode simulations translate with the same absorber on both sides of a
# zero-size axis folded into ``Periodic``.
ZERO_SIZE_AXIS_FOLD = (
    "flexcompute.core._migration.solver.fdtd.conversion.boundaries:fold_boundaries_for_zero_dims"
)

# Scopes that check every subclass, whatever family it belongs to, before the subclass's
# own scope. Each facade takes the core the model keeps, and these form no family: every
# geometry, including each Box subclass such as a monitor or a source, checks its core.
INHERITED_SCOPES: dict[str, str] = {
    "tidy3d.components.geometry.base.core:Geometry": "flexcompute.geometry.validation:validate",
}

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
    # Off the construction path for performance: a 3-pole fit copies ``FastFitterData`` about 2000
    # times, and translating each copy makes the fit 46% slower (1.19 s to 1.74 s). The client keeps
    # a field adapter and a thin model check that call the core rules.
    "tidy3d.components.dispersion_fitter:AdvancedFastFitterParam": ComponentScope(
        "flexcompute.core._migration.em.material.dispersion_fitter.schema.dispersion_fitter:AdvancedFastFitterParam",
        "flexcompute.core._migration.em.material.dispersion_fitter.validation:validate",
    ),
    "tidy3d.components.dispersion_fitter:FastFitterData": ComponentScope(
        "flexcompute.core._migration.em.material.dispersion_fitter.schema.dispersion_fitter:FastFitterData",
        "flexcompute.core._migration.em.material.dispersion_fitter.validation:validate",
    ),
    "tidy3d.components.parameter_perturbation:ParameterPerturbation": ComponentScope(
        "flexcompute.core._migration.em.material.parameter_perturbation.schema.parameter_perturbation:ParameterPerturbation",
        "flexcompute.core._migration.em.material.parameter_perturbation.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.parameter_perturbation:PermittivityPerturbation": ComponentScope(
        "flexcompute.core._migration.em.material.parameter_perturbation.schema.parameter_perturbation:PermittivityPerturbation",
        "flexcompute.core._migration.em.material.parameter_perturbation.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.parameter_perturbation:IndexPerturbation": ComponentScope(
        "flexcompute.core._migration.em.material.parameter_perturbation.schema.parameter_perturbation:IndexPerturbation",
        "flexcompute.core._migration.em.material.parameter_perturbation.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.parameter_perturbation:ElectroOpticPerturbation": ComponentScope(
        "flexcompute.core._migration.em.material.parameter_perturbation.schema.parameter_perturbation:ElectroOpticPerturbation",
        "flexcompute.core._migration.em.material.parameter_perturbation.validation:validate",
        on_construction=True,
    ),
    # The abstract roots have no schema. The concrete classes below carry no rules, so their
    # facade reports nothing.
    "tidy3d.components.parameter_perturbation:AbstractPerturbation": None,
    "tidy3d.components.parameter_perturbation:HeatPerturbation": None,
    "tidy3d.components.parameter_perturbation:ChargePerturbation": None,
    "tidy3d.components.parameter_perturbation:AbstractDeltaModel": None,
    "tidy3d.components.parameter_perturbation:LinearHeatPerturbation": ComponentScope(
        "flexcompute.core._migration.em.material.parameter_perturbation.schema.parameter_perturbation:LinearHeatPerturbation",
        "flexcompute.core._migration.em.material.parameter_perturbation.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.parameter_perturbation:CustomHeatPerturbation": ComponentScope(
        "flexcompute.core._migration.em.material.parameter_perturbation.schema.parameter_perturbation:CustomHeatPerturbation",
        "flexcompute.core._migration.em.material.parameter_perturbation.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.parameter_perturbation:LinearChargePerturbation": ComponentScope(
        "flexcompute.core._migration.em.material.parameter_perturbation.schema.parameter_perturbation:LinearChargePerturbation",
        "flexcompute.core._migration.em.material.parameter_perturbation.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.parameter_perturbation:CustomChargePerturbation": ComponentScope(
        "flexcompute.core._migration.em.material.parameter_perturbation.schema.parameter_perturbation:CustomChargePerturbation",
        "flexcompute.core._migration.em.material.parameter_perturbation.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.parameter_perturbation:NedeljkovicSorefMashanovich": ComponentScope(
        "flexcompute.core._migration.em.material.parameter_perturbation.schema.parameter_perturbation:NedeljkovicSorefMashanovich",
        "flexcompute.core._migration.em.material.parameter_perturbation.validation:validate",
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
    "tidy3d.components.medium.lossy_metal:LossyMetalMedium": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.lossy_metal:LossyMetalMedium",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
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
    # ``import tidy3d`` builds Medium2D, so it stays off the construction path.
    "tidy3d.components.medium.two_d:Medium2D": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.two_d:Medium2D",
        "flexcompute.core._migration.em.material.medium.validation:validate",
    ),
    "tidy3d.components.medium.custom.isotropic:CustomIsotropicMedium": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.custom.isotropic:CustomIsotropicMedium",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.medium.custom.isotropic:CustomMedium": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.custom.isotropic:CustomMedium",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    # Abstract family roots with no schema: a Tidy3D subclass that misses its own registration fails,
    # and one defined elsewhere opts out.
    "tidy3d.components.medium.pole_residue:DispersiveMedium": None,
    "tidy3d.components.medium.abstract_custom:AbstractCustomMedium": None,
    "tidy3d.components.medium.custom.dispersive:CustomDispersiveMedium": None,
    "tidy3d.components.medium.perturbation:AbstractPerturbationMedium": None,
    "tidy3d.components.medium.perturbation:AbstractIsotropicPerturbationMedium": None,
    "tidy3d.components.medium.drude:Drude": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.drude:Drude",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.medium.debye:Debye": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.debye:Debye",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.medium.lorentz:Lorentz": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.lorentz:Lorentz",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    # ``import tidy3d`` builds Sellmeier, so it stays off the construction path.
    "tidy3d.components.medium.sellmeier:Sellmeier": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.sellmeier:Sellmeier",
        "flexcompute.core._migration.em.material.medium.validation:validate",
    ),
    "tidy3d.components.medium.custom.dispersive:CustomPoleResidue": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.custom.dispersive:CustomPoleResidue",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.medium.custom.dispersive:CustomSellmeier": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.custom.dispersive:CustomSellmeier",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.medium.custom.dispersive:CustomLorentz": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.custom.dispersive:CustomLorentz",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.medium.custom.dispersive:CustomDrude": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.custom.dispersive:CustomDrude",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.medium.custom.dispersive:CustomDebye": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.custom.dispersive:CustomDebye",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.medium.anisotropic:AnisotropicMedium": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.anisotropic:AnisotropicMedium",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.medium.anisotropic:AnisotropicMediumFromMedium2D": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.anisotropic:AnisotropicMediumFromMedium2D",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.medium.custom.anisotropic:CustomAnisotropicMedium": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.custom.anisotropic:CustomAnisotropicMedium",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.medium.custom.anisotropic:CustomAnisotropicMediumInternal": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.custom.anisotropic:CustomAnisotropicMediumInternal",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    # ``import tidy3d`` builds these, so they stay off the construction path. Their thin
    # client checks call the core rules.
    "tidy3d.components.medium.isotropic:Medium": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.isotropic:Medium",
        "flexcompute.core._migration.em.material.medium.validation:validate",
    ),
    "tidy3d.components.medium.pole_residue:PoleResidue": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.pole_residue:PoleResidue",
        "flexcompute.core._migration.em.material.medium.validation:validate",
    ),
    "tidy3d.components.medium.perturbation:PerturbationMedium": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.perturbation:PerturbationMedium",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.medium.perturbation:PerturbationPoleResidue": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.perturbation:PerturbationPoleResidue",
        "flexcompute.core._migration.em.material.medium.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.medium.perturbation:PerturbationAnisotropicMedium": ComponentScope(
        "flexcompute.core._migration.em.material.medium.schema.perturbation:PerturbationAnisotropicMedium",
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
    "tidy3d.components.tcad.ionization:PearsonBardeenIonization": ComponentScope(
        "flexcompute.core._migration.tcad.schema.ionization:PearsonBardeenIonization",
        "flexcompute.core._migration.tcad.validation:validate",
    ),
    "tidy3d.components.tcad.ionization:AltermattSchenkIonization": ComponentScope(
        "flexcompute.core._migration.tcad.schema.ionization:AltermattSchenkIonization",
        "flexcompute.core._migration.tcad.validation:validate",
    ),
    "tidy3d.components.tcad.ionization:IonizationSite": ComponentScope(
        "flexcompute.core._migration.tcad.schema.ionization:IonizationSite",
        "flexcompute.core._migration.tcad.validation:validate",
    ),
    "tidy3d.components.tcad.ionization:MultiSiteIonization": ComponentScope(
        "flexcompute.core._migration.tcad.schema.ionization:MultiSiteIonization",
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
    "tidy3d.components.monitor:FieldStructureTimeMonitor": ComponentScope(
        "flexcompute.core._migration.em.monitor.schema:FieldStructureTimeMonitor",
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
    "tidy3d.components.tcad.doping:ProtectedDopingRef": ComponentScope(
        "flexcompute.core._migration.tcad.schema.doping:ProtectedDopingRef",
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
    # scene: transformation and structure
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
    # Translation is too slow for the construction path: a simulation rebuilds every
    # structure, and grid generation builds many internal overrides. ``Structure`` keeps its
    # checks in the schema; ``MeshOverrideStructure`` keeps thin checks.
    "tidy3d.components.structure:AbstractStructure": None,
    "tidy3d.components.structure:Structure": ComponentScope(
        "flexcompute.core._migration.em.structure.schema:Structure",
        "flexcompute.core._migration.em.structure.validation:validate",
    ),
    "tidy3d.components.structure:MeshOverrideStructure": ComponentScope(
        "flexcompute.core._migration.em.structure.schema:MeshOverrideStructure",
        "flexcompute.core._migration.em.structure.validation:validate",
    ),
    # A simulation builds its scene on demand, so ``Scene`` stays off the construction path and
    # keeps thin checks. Registered for the simulation scopes that will run its rules.
    "tidy3d.components.scene:Scene": ComponentScope(
        "flexcompute.core._migration.em.scene.schema:Scene",
        "flexcompute.core._migration.em.scene.validation:validate",
    ),
    # source
    "tidy3d.components.source.current:UniformCurrentSource": ComponentScope(
        "flexcompute.core._migration.em.source.schema.current:UniformCurrentSource",
        "flexcompute.core._migration.em.source.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.source.current:PointDipole": ComponentScope(
        "flexcompute.core._migration.em.source.schema.current:PointDipole",
        "flexcompute.core._migration.em.source.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.source.current:CustomCurrentSource": ComponentScope(
        "flexcompute.core._migration.em.source.schema.current:CustomCurrentSource",
        "flexcompute.core._migration.em.source.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.source.field:CustomFieldSource": ComponentScope(
        "flexcompute.core._migration.em.source.schema.field:CustomFieldSource",
        "flexcompute.core._migration.em.source.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.source.field:ModeSource": ComponentScope(
        "flexcompute.core._migration.em.source.schema.field:ModeSource",
        "flexcompute.core._migration.em.source.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.source.field:PlaneWave": ComponentScope(
        "flexcompute.core._migration.em.source.schema.field:PlaneWave",
        "flexcompute.core._migration.em.source.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.source.field:GaussianBeam": ComponentScope(
        "flexcompute.core._migration.em.source.schema.field:GaussianBeam",
        "flexcompute.core._migration.em.source.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.source.field:AstigmaticGaussianBeam": ComponentScope(
        "flexcompute.core._migration.em.source.schema.field:AstigmaticGaussianBeam",
        "flexcompute.core._migration.em.source.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.source.field:ThinLensBeam": ComponentScope(
        "flexcompute.core._migration.em.source.schema.field:ThinLensBeam",
        "flexcompute.core._migration.em.source.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.source.field:TFSF": ComponentScope(
        "flexcompute.core._migration.em.source.schema.field:TFSF",
        "flexcompute.core._migration.em.source.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.source.time:GaussianPulse": ComponentScope(
        "flexcompute.core._migration.em.source.schema.time:GaussianPulse",
        "flexcompute.core._migration.em.source.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.source.time:ContinuousWave": ComponentScope(
        "flexcompute.core._migration.em.source.schema.time:ContinuousWave",
        "flexcompute.core._migration.em.source.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.source.time:CustomSourceTime": ComponentScope(
        "flexcompute.core._migration.em.source.schema.time:CustomSourceTime",
        "flexcompute.core._migration.em.source.validation:validate",
        on_construction=True,
    ),
    # The schema evaluates the pulse from the Gaussian pulses the translation reads from
    # ``tidy3d-extras``.
    "tidy3d.components.source.time:BroadbandPulse": ComponentScope(
        "flexcompute.core._migration.em.source.schema.time:BroadbandPulse",
        "flexcompute.core._migration.em.source.validation:validate",
        on_construction=True,
        translator="tidy3d.em.translate.source:fill_broadband_pulse",
    ),
    # mode: ``import tidy3d`` builds ModeSortSpec, ModeInterpSpec and ChebSampling objects (the
    # EMEModeSpec default), so they stay off the construction path.
    "tidy3d.components.mode_spec:ModeSortSpec": ComponentScope(
        "flexcompute.core._migration.em.mode.spec.schema:ModeSortSpec",
        "flexcompute.core._migration.em.mode.spec.validation:validate",
    ),
    "tidy3d.components.mode_spec:FrequencySamplingSpec": None,
    "tidy3d.components.mode_spec:UniformSampling": ComponentScope(
        "flexcompute.core._migration.em.mode.spec.schema:UniformSampling",
        "flexcompute.core._migration.em.mode.spec.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.mode_spec:ChebSampling": ComponentScope(
        "flexcompute.core._migration.em.mode.spec.schema:ChebSampling",
        "flexcompute.core._migration.em.mode.spec.validation:validate",
    ),
    "tidy3d.components.mode_spec:CustomSampling": ComponentScope(
        "flexcompute.core._migration.em.mode.spec.schema:CustomSampling",
        "flexcompute.core._migration.em.mode.spec.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.mode_spec:ModeInterpSpec": ComponentScope(
        "flexcompute.core._migration.em.mode.spec.schema:ModeInterpSpec",
        "flexcompute.core._migration.em.mode.spec.validation:validate",
    ),
    # boundary: placement
    "tidy3d.components.bc_placement:AbstractBCPlacement": None,
    "tidy3d.components.bc_placement:StructureBoundary": ComponentScope(
        "flexcompute.core._migration.em.boundary.placement.schema:StructureBoundary",
        "flexcompute.core._migration.em.boundary.placement.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.bc_placement:StructureStructureInterface": ComponentScope(
        "flexcompute.core._migration.em.boundary.placement.schema:StructureStructureInterface",
        "flexcompute.core._migration.em.boundary.placement.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.bc_placement:MediumMediumInterface": ComponentScope(
        "flexcompute.core._migration.em.boundary.placement.schema:MediumMediumInterface",
        "flexcompute.core._migration.em.boundary.placement.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.bc_placement:SimulationBoundary": ComponentScope(
        "flexcompute.core._migration.em.boundary.placement.schema:SimulationBoundary",
        "flexcompute.core._migration.em.boundary.placement.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.bc_placement:StructureSimulationBoundary": ComponentScope(
        "flexcompute.core._migration.em.boundary.placement.schema:StructureSimulationBoundary",
        "flexcompute.core._migration.em.boundary.placement.validation:validate",
        on_construction=True,
    ),
    # boundary: boundary conditions
    "tidy3d.components.boundary:BoundaryEdge": None,
    "tidy3d.components.boundary:AbstractABCBoundary": None,
    "tidy3d.components.boundary:ABCBoundary": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:ABCBoundary",
        "flexcompute.core._migration.em.boundary.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.boundary:BroadbandModeABCSpec": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:BroadbandModeABCSpec",
        "flexcompute.core._migration.em.boundary.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.boundary:ModeABCBoundary": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:ModeABCBoundary",
        "flexcompute.core._migration.em.boundary.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.boundary:InternalAbsorber": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:InternalAbsorber",
        "flexcompute.core._migration.em.boundary.validation:validate",
        on_construction=True,
    ),
    # Built at import as the default fit parameters, so off the construction path; it has no rule.
    "tidy3d.components.boundary:BroadbandModeABCFitterParam": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:BroadbandModeABCFitterParam",
        "flexcompute.core._migration.em.boundary.validation:validate",
    ),
    # Built at import by the BoundarySpec docstring and the default parameters, so these stay off
    # the construction path; Boundary runs its rules from a thin check.
    "tidy3d.components.boundary:Boundary": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:Boundary",
        "flexcompute.core._migration.em.boundary.validation:validate",
    ),
    "tidy3d.components.boundary:BoundarySpec": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:BoundarySpec",
        "flexcompute.core._migration.em.boundary.validation:validate",
    ),
    "tidy3d.components.boundary:PML": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:PML",
        "flexcompute.core._migration.em.boundary.validation:validate",
    ),
    "tidy3d.components.boundary:PECBoundary": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:PECBoundary",
        "flexcompute.core._migration.em.boundary.validation:validate",
    ),
    "tidy3d.components.boundary:PMLParams": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:PMLParams",
        "flexcompute.core._migration.em.boundary.validation:validate",
    ),
    "tidy3d.components.boundary:AbsorberParams": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:AbsorberParams",
        "flexcompute.core._migration.em.boundary.validation:validate",
    ),
    "tidy3d.components.boundary:Periodic": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:Periodic",
        "flexcompute.core._migration.em.boundary.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.boundary:PMCBoundary": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:PMCBoundary",
        "flexcompute.core._migration.em.boundary.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.boundary:BlochBoundary": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:BlochBoundary",
        "flexcompute.core._migration.em.boundary.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.boundary:AbsorberSpec": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:AbsorberSpec",
        "flexcompute.core._migration.em.boundary.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.boundary:StablePML": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:StablePML",
        "flexcompute.core._migration.em.boundary.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.boundary:Absorber": ComponentScope(
        "flexcompute.core._migration.em.boundary.schema:Absorber",
        "flexcompute.core._migration.em.boundary.validation:validate",
        on_construction=True,
    ),
    # heat-charge
    "tidy3d.components.tcad.analysis.heat_simulation_type:UnsteadySpec": ComponentScope(
        "flexcompute.core._migration.solver.heat_charge.schema.heat_simulation_type:UnsteadySpec",
        "flexcompute.core._migration.solver.heat_charge.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.analysis.heat_simulation_type:UnsteadyHeatAnalysis": ComponentScope(
        "flexcompute.core._migration.solver.heat_charge.schema.heat_simulation_type:UnsteadyHeatAnalysis",
        "flexcompute.core._migration.solver.heat_charge.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.spice.analysis.ac:AbstractSSACAnalysis": None,
    # Built at import as the default `tolerance_settings`, so off the construction path; no rule.
    "tidy3d.components.spice.analysis.dc:ChargeToleranceSpec": ComponentScope(
        "flexcompute.core._migration.solver.heat_charge.schema.dc:ChargeToleranceSpec",
        "flexcompute.core._migration.solver.heat_charge.validation:validate",
    ),
    "tidy3d.components.spice.analysis.dc:SteadyChargeDCAnalysis": ComponentScope(
        "flexcompute.core._migration.solver.heat_charge.schema.dc:SteadyChargeDCAnalysis",
        "flexcompute.core._migration.solver.heat_charge.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.spice.analysis.dc:IsothermalSteadyChargeDCAnalysis": ComponentScope(
        "flexcompute.core._migration.solver.heat_charge.schema.dc:IsothermalSteadyChargeDCAnalysis",
        "flexcompute.core._migration.solver.heat_charge.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.spice.analysis.ac:SSACAnalysis": ComponentScope(
        "flexcompute.core._migration.solver.heat_charge.schema.ac:SSACAnalysis",
        "flexcompute.core._migration.solver.heat_charge.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.spice.analysis.ac:IsothermalSSACAnalysis": ComponentScope(
        "flexcompute.core._migration.solver.heat_charge.schema.ac:IsothermalSSACAnalysis",
        "flexcompute.core._migration.solver.heat_charge.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.simulation.heat_charge:HeatChargeSimulation": ComponentScope(
        "flexcompute.core._migration.solver.heat_charge.schema.heat_charge:HeatChargeSimulation",
        "flexcompute.core._migration.solver.heat_charge.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.tcad.simulation.heat:HeatSimulation": ComponentScope(
        "flexcompute.core._migration.solver.heat_charge.schema.heat:HeatSimulation",
        "flexcompute.core._migration.solver.heat_charge.validation:validate",
        on_construction=True,
    ),
    # fast-heat
    "tidy3d.components.fast_heat.simulation:FastHeatSimulation": ComponentScope(
        "flexcompute.core._migration.solver.fast_heat.schema.fast_heat:FastHeatSimulation",
        "flexcompute.core._migration.solver.fast_heat.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.fast_heat.grid:FastHeatGridSpec": ComponentScope(
        "flexcompute.core._migration.solver.fast_heat.schema.grid:FastHeatGridSpec",
        "flexcompute.core._migration.solver.fast_heat.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.fast_heat.grid:FastHeatLayerSpec": ComponentScope(
        "flexcompute.core._migration.solver.fast_heat.schema.grid:FastHeatLayerSpec",
        "flexcompute.core._migration.solver.fast_heat.validation:validate",
        on_construction=True,
    ),
    # eme-workflow
    "tidy3d.components.eme.simulation:EMESimulation": ComponentScope(
        "flexcompute.core._migration.solver.eme.schema.simulation:EMESimulation",
        "flexcompute.core._migration.solver.eme.validation:validate",
        on_construction=True,
        translator=ZERO_SIZE_AXIS_FOLD,
    ),
    # fdtd
    "tidy3d.components.simulation.model:Simulation": ComponentScope(
        "flexcompute.core._migration.solver.fdtd.schema.model:Simulation",
        "flexcompute.core._migration.solver.fdtd.validation:validate",
        on_construction=True,
        translator=ZERO_SIZE_AXIS_FOLD,
    ),
    # mode-workflow
    "tidy3d.components.mode.mode_solver:ModeSolver": ComponentScope(
        "flexcompute.core._migration.solver.mode.schema.mode_solver:ModeSolver",
        "flexcompute.core._migration.solver.mode.validation:validate",
        on_construction=True,
    ),
    "tidy3d.components.mode.simulation:ModeSimulation": ComponentScope(
        "flexcompute.core._migration.solver.mode.schema.simulation:ModeSimulation",
        "flexcompute.core._migration.solver.mode.validation:validate",
        on_construction=True,
        translator=ZERO_SIZE_AXIS_FOLD,
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
    "em.material.medium.nonfinite_frequency_range": "tidy3d.exceptions:ValidationError",
    "em.material.medium.nonpositive_frequency_range": "tidy3d.exceptions:ValidationError",
    "em.material.medium.anisotropic_surface_lossy_metal_component": "tidy3d.exceptions:ValidationError",
    "em.material.medium.custom_anisotropic_component_not_isotropic": "tidy3d.exceptions:SetupError",
    "em.material.medium.perturbation_shape_mismatch": "tidy3d.exceptions:SetupError",
    "em.material.medium.perturbation_complex": "tidy3d.exceptions:SetupError",
    "em.material.medium.permittivity_component_single_frequency": "tidy3d.exceptions:SetupError",
    "em.material.medium.custom_isotropic_permittivity_real": "tidy3d.exceptions:SetupError",
    "em.material.medium.custom_isotropic_permittivity_ge_one": "tidy3d.exceptions:SetupError",
    "em.material.medium.custom_isotropic_conductivity_real": "tidy3d.exceptions:SetupError",
    "em.material.medium.custom_isotropic_conductivity_coordinates": "tidy3d.exceptions:SetupError",
    "em.material.medium.custom_isotropic_passive_conductivity_nonnegative": "tidy3d.exceptions:ValidationError",
    "em.material.medium.custom_eps_inf_real": "tidy3d.exceptions:SetupError",
    "em.material.medium.custom_eps_inf_nonnegative": "tidy3d.exceptions:SetupError",
    "em.material.medium.custom_coeffs_coordinates": "tidy3d.exceptions:SetupError",
    "em.material.medium.custom_coeffs_real": "tidy3d.exceptions:SetupError",
    "em.material.medium.custom_lorentz_delta_nonnegative": "tidy3d.exceptions:ValidationError",
    "em.material.medium.custom_debye_tau_minimum": "tidy3d.exceptions:SetupError",
    "em.material.medium.custom_sellmeier_coordinates": "tidy3d.exceptions:SetupError",
    "em.material.medium.custom_sellmeier_real": "tidy3d.exceptions:SetupError",
    "em.material.medium.custom_sellmeier_c_positive": "tidy3d.exceptions:SetupError",
    "em.material.medium.custom_sellmeier_c_near_zero": "tidy3d.exceptions:SetupError",
    "em.material.medium.custom_lorentz_delta_ordering": "tidy3d.exceptions:SetupError",
    "em.material.medium.custom_passive_delta_epsilon_nonnegative": "tidy3d.exceptions:ValidationError",
    "em.material.medium.custom_passive_sellmeier_b_nonnegative": "tidy3d.exceptions:ValidationError",
    "em.material.medium.pole_residue_causal": "tidy3d.exceptions:SetupError",
    "em.material.medium.pole_a_too_large": "tidy3d.exceptions:ValidationError",
    "em.material.medium.pole_c_too_large": "tidy3d.exceptions:ValidationError",
    "em.material.medium.lorentz_f_delta_distinct": "tidy3d.exceptions:SetupError",
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
    # scene: structure
    "em.structure.non_finite_min_steps_per_size": "builtins:ValueError",
    # scene: media
    "em.scene.too_many_mediums": "tidy3d.exceptions:SetupError",
    "em.scene.too_many_structures_per_medium": "tidy3d.exceptions:SetupError",
    # source
    "em.source.too_few_time_coords": "tidy3d.exceptions:ValidationError",
    "em.source.non_increasing_time_coords": "tidy3d.exceptions:ValidationError",
    "em.source.nan_dataset": "tidy3d.exceptions:ValidationError",
    "em.source.duplicate_dataset_coords": "tidy3d.exceptions:DataError",
    "em.source.non_positive_freq_range": "tidy3d.exceptions:ValidationError",
    "em.source.unordered_freq_range": "tidy3d.exceptions:ValidationError",
    # mode
    "em.mode.spec.custom_sampling_min_points": "tidy3d.exceptions:ValidationError",
    "em.mode.spec.custom_sampling_sorted": "tidy3d.exceptions:ValidationError",
    # boundary: placement
    "em.boundary.placement.duplicate_structure": "tidy3d.exceptions:SetupError",
    "em.boundary.placement.duplicate_medium": "tidy3d.exceptions:SetupError",
    # boundary: boundary conditions
    "em.boundary.frequency_range_order": "tidy3d.exceptions:ValidationError",
    "em.boundary.mode_abc_plane_not_planar": "tidy3d.exceptions:ValidationError",
    "em.boundary.internal_absorber_permittivity": "tidy3d.exceptions:ValidationError",
    # heat-charge
    "solver.heat_charge.ssac_freqs_empty": "builtins:ValueError",
    "solver.heat_charge.ssac_freqs_infinite": "builtins:ValueError",
    "solver.heat_charge.ssac_freqs_negative": "builtins:ValueError",
    # fast-heat: the same exception type as the matching HeatChargeSimulation check
    "solver.fast_heat.zero_dims": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.infinite_domain": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.symmetry_not_supported": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.grid_axis_on_zero_dim": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.grid_axis_missing": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.layer_boundary_outside_domain": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.custom_boundaries_do_not_span_domain": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.unsupported_boundary_condition": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.natural_convection_not_supported": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.emissivity_not_supported": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.bc_on_zero_axis": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.unknown_structure": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.unknown_medium": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.invalid_background_medium": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.temperature_dependent_property": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.rotated_conductivity": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.advection_not_supported": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.non_finite_source_rate": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.non_finite_value": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.complex_source_rate": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.spatial_rate_bounds": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.extruded_source_rate": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.extruded_initial_temperature": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.no_layer_boundaries": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.layer_boundaries_not_increasing": "tidy3d.exceptions:SetupError",
    "solver.fast_heat.layer_grid_spec_count": "tidy3d.exceptions:SetupError",
    # eme-workflow
    "solver.eme.duplicate_freqs": "tidy3d.exceptions:SetupError",
    "solver.eme.non_reciprocal_medium": "tidy3d.exceptions:SetupError",
    # fdtd
    "solver.fdtd.auto_grid_wavelength_undefined": "tidy3d.exceptions:SetupError",
    "solver.fdtd.incompatible_material_intersection": "tidy3d.exceptions:ValidationError",
    "solver.fdtd.microwave_auto_impedance_setup": "tidy3d.exceptions:SetupError",
    "solver.fdtd.no_sources": "tidy3d.exceptions:SetupError",
    "solver.fdtd.too_many_sources": "tidy3d.exceptions:SetupError",
    # mode-workflow
    "solver.mode.angle_rotation_unsupported_medium": "tidy3d.exceptions:SetupError",
    "solver.mode.angle_rotation_structure_failed": "tidy3d.exceptions:SetupError",
    # geometry: the scope checks the fields the geometry classes once checked themselves, and
    # raises each error as their field check did
    "geometry.center_infinite": "tidy3d.exceptions:ValidationError",
    "geometry.radius_infinite": "tidy3d.exceptions:ValidationError",
    "geometry.sidewall_angle_out_of_range": "tidy3d.exceptions:ValidationError",
    "geometry.polyslab.slab_bounds_order": "tidy3d.exceptions:SetupError",
    "geometry.polyslab.vertices_shape": "tidy3d.exceptions:SetupError",
    "geometry.transform.singular": "tidy3d.exceptions:ValidationError",
    "geometry.transform.projective": "tidy3d.exceptions:AdjointError",
    "geometry.array.transform_translates": "tidy3d.exceptions:ValidationError",
    # geometry: the scope reports an unbounded geometry that a Transformed or GeometryArray places
    "geometry.unbounded_geometry": "tidy3d.exceptions:ValidationError",
}

# Warnings that Tidy3D logs once per session, as their client checks did.
COMPONENT_WARNINGS_LOGGED_ONCE = frozenset(
    {
        "solver.fdtd.fixed_angle_tfsf_normal_incidence",
        "solver.fdtd.fixed_angle_tfsf_long_run_time",
    }
)


def resolve(path: str) -> Any:
    """Resolve a ``module:attribute`` path lazily."""

    module_name, attr = path.split(":", 1)
    return getattr(import_module(module_name), attr)

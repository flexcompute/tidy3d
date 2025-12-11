"""Workflow definition classes for multi-step simulations.

This module provides the core abstractions for defining execution workflows:
- `Step`: A single unit of work with typed inputs and outputs
- `StepInput`/`StepOutput`: Schema classes for step I/O
- `Workflow`: A DAG of steps that can be executed together
- `HeatChargeWorkflow`: Pre-defined workflow for HeatCharge simulations
"""

from __future__ import annotations

from typing import Optional, Union

import pydantic.v1 as pd

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.tcad.mesher import VolumeMesher
from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation
from tidy3d.components.types.workflow import WorkflowType

# =============================================================================
# Step Input/Output Schema Classes
# =============================================================================


class StepInput(Tidy3dBaseModel):
    """Base class for step inputs.

    Defines what data a step requires from a previous step in the workflow.
    Subclasses define the specific type of data (mesh, field data, etc.).
    Type discrimination is handled automatically via Tidy3dBaseModel's `type` field.
    """

    source_step: str = pd.Field(
        ...,
        title="Source Step",
        description="Name of the step that produces this input.",
    )

    required: bool = pd.Field(
        True,
        title="Required",
        description="Whether this input is required for the step to run.",
    )


class MeshInput(StepInput):
    """Input: mesh data from a meshing step."""


class FieldDataInput(StepInput):
    """Input: field data from a solver step."""


class HeatSourceInput(StepInput):
    """Input: heat source distribution for thermal solver."""


class StepOutput(Tidy3dBaseModel):
    """Base class for step outputs.

    Defines what data a step produces for subsequent steps (pipeline data).
    This is separate from user-requested monitor data.
    Type discrimination is handled automatically via Tidy3dBaseModel's `type` field.
    """


class MeshOutput(StepOutput):
    """Output: mesh data for subsequent steps."""


class SimulationDataOutput(StepOutput):
    """Output: simulation results."""

    # Note: We store the class name as a string to avoid circular imports
    # and serialization issues with type objects
    data_type_name: str = pd.Field(
        ...,
        title="Data Type Name",
        description="Name of the expected data type (e.g., 'HeatChargeSimulationData').",
    )


class FieldDataOutput(StepOutput):
    """Output: field data for subsequent steps."""


class HeatSourceOutput(StepOutput):
    """Output: heat source data for thermal solver."""


# Union of all input/output types for validation
StepInputType = Union[MeshInput, FieldDataInput, HeatSourceInput]
StepOutputType = Union[MeshOutput, SimulationDataOutput, FieldDataOutput, HeatSourceOutput]

# Mapping from input class to compatible output class(es)
INPUT_OUTPUT_COMPATIBILITY: dict[type[StepInput], tuple[type[StepOutput], ...]] = {
    MeshInput: (MeshOutput,),
    FieldDataInput: (FieldDataOutput, SimulationDataOutput),
    HeatSourceInput: (HeatSourceOutput,),
}


# =============================================================================
# Step Class
# =============================================================================


class Step(Tidy3dBaseModel):
    """A single step in a workflow.

    Each step represents a unit of work (simulation, meshing, conversion)
    with explicit typed inputs and outputs.
    """

    name: str = pd.Field(
        ...,
        title="Name",
        description="Unique name for this step within the workflow.",
    )

    simulation: WorkflowType = pd.Field(
        ...,
        title="Simulation",
        description="The simulation or operation to execute in this step.",
        discriminator="type",
    )

    allow_async: bool = pd.Field(
        True,
        title="Allow Async",
        description=(
            "Whether this step may run in parallel with other async-allowed steps. "
            "Data dependencies automatically enforce ordering regardless of this flag."
        ),
    )

    inputs: tuple[StepInputType, ...] = pd.Field(
        (),
        title="Inputs",
        description="List of inputs required from previous steps.",
    )

    outputs: tuple[StepOutputType, ...] = pd.Field(
        (),
        title="Outputs",
        description="List of outputs produced by this step for subsequent steps.",
    )

    @property
    def input_types(self) -> set[type[StepInput]]:
        """Return set of input types."""
        return {type(inp) for inp in self.inputs}

    @property
    def output_types(self) -> set[type[StepOutput]]:
        """Return set of output types."""
        return {type(out) for out in self.outputs}

    @property
    def source_steps(self) -> set[str]:
        """Return set of step names this step depends on."""
        return {inp.source_step for inp in self.inputs}


# =============================================================================
# Workflow Class
# =============================================================================


class Workflow(Tidy3dBaseModel):
    """A complete workflow definition as a DAG of steps.

    The workflow is validated at construction time to ensure:
    - No duplicate step names
    - All input references point to earlier steps
    - Input types match output types from source steps

    This class is serializable to JSON for persistence and debugging.
    """

    steps: tuple[Step, ...] = pd.Field(
        ...,
        title="Steps",
        description="Ordered list of steps in the workflow.",
    )

    @pd.validator("steps")
    def _validate_dag(cls, steps: tuple[Step, ...]) -> tuple[Step, ...]:
        """Validate the workflow DAG structure."""
        if not steps:
            raise ValueError("Workflow must have at least one step")

        step_names: set[str] = set()
        outputs_by_step: dict[str, set[type[StepOutput]]] = {}

        for step in steps:
            # Check for duplicate names
            if step.name in step_names:
                raise ValueError(f"Duplicate step name: '{step.name}'")
            step_names.add(step.name)

            # Validate inputs reference existing earlier steps
            for inp in step.inputs:
                if inp.source_step not in step_names:
                    if inp.source_step in {s.name for s in steps}:
                        raise ValueError(
                            f"Step '{step.name}' depends on '{inp.source_step}' "
                            "which comes later in the workflow"
                        )
                    else:
                        raise ValueError(
                            f"Step '{step.name}' references unknown step '{inp.source_step}'"
                        )

                # Check that source step produces a compatible output type
                source_outputs = outputs_by_step.get(inp.source_step, set())
                input_type = type(inp)
                compatible_outputs = INPUT_OUTPUT_COMPATIBILITY.get(input_type, ())
                if not any(out_type in source_outputs for out_type in compatible_outputs):
                    source_output_names = {t.__name__ for t in source_outputs} or {"nothing"}
                    raise ValueError(
                        f"Step '{step.name}' expects {input_type.__name__} from "
                        f"'{inp.source_step}', but that step produces: {source_output_names}"
                    )

            # Record this step's outputs
            outputs_by_step[step.name] = step.output_types

        return steps

    @property
    def num_steps(self) -> int:
        """Return the number of steps in this workflow."""
        return len(self.steps)

    @property
    def is_single_step(self) -> bool:
        """Return True if this is a single-step workflow."""
        return self.num_steps == 1

    @property
    def step_names(self) -> list[str]:
        """Return list of step names in order."""
        return [step.name for step in self.steps]

    def get_step(self, name: str) -> Optional[Step]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def get_step_index(self, name: str) -> Optional[int]:
        """Get the index of a step by name."""
        for i, step in enumerate(self.steps):
            if step.name == name:
                return i
        return None

    def execution_order(self) -> list[list[str]]:
        """Return steps grouped by execution phase.

        Steps that can run in parallel (no dependencies between them)
        are grouped together. This is for Phase 2+ parallel execution.

        Returns
        -------
        list[list[str]]
            List of phases, each containing step names that can run in parallel.
        """
        # For now, return sequential execution (one step per phase)
        # Phase 2 will implement proper parallel grouping based on dependencies
        return [[step.name] for step in self.steps]


# =============================================================================
# Predefined Workflow Classes
# =============================================================================


class HeatChargeWorkflow(Workflow):
    """Workflow for HeatCharge simulations: mesh → solve.

    This is the standard workflow for HeatChargeSimulation, consisting of:
    1. A meshing step using VolumeMesher
    2. A solve step using the HeatChargeSimulation
    """

    @pd.validator("steps")
    def _validate_heat_charge_structure(cls, steps: tuple[Step, ...]) -> tuple[Step, ...]:
        """Validate HeatCharge-specific workflow structure."""
        # Must have exactly 2 steps
        if len(steps) != 2:
            raise ValueError(f"HeatChargeWorkflow must have exactly 2 steps, got {len(steps)}")

        mesh_step, solve_step = steps

        # Validate step names
        if mesh_step.name != "mesh":
            raise ValueError(f"First step must be named 'mesh', got '{mesh_step.name}'")
        if solve_step.name != "solve":
            raise ValueError(f"Second step must be named 'solve', got '{solve_step.name}'")

        # Validate simulation types
        if not isinstance(mesh_step.simulation, VolumeMesher):
            raise ValueError(
                f"First step must contain a VolumeMesher, got {type(mesh_step.simulation).__name__}"
            )
        if not isinstance(solve_step.simulation, HeatChargeSimulation):
            raise ValueError(
                f"Second step must contain a HeatChargeSimulation, "
                f"got {type(solve_step.simulation).__name__}"
            )

        # Validate the mesher references the same simulation as the solve step
        if mesh_step.simulation.simulation is not solve_step.simulation:
            raise ValueError(
                "The VolumeMesher must reference the same HeatChargeSimulation as the solve step"
            )

        return steps

    @classmethod
    def from_simulation(cls, simulation: HeatChargeSimulation) -> HeatChargeWorkflow:
        """Construct workflow from a HeatChargeSimulation.

        Parameters
        ----------
        simulation : HeatChargeSimulation
            The simulation to create a workflow for.

        Returns
        -------
        HeatChargeWorkflow
            A workflow with mesh and solve steps.
        """
        return cls(
            steps=(
                Step(
                    name="mesh",
                    simulation=VolumeMesher(simulation=simulation),
                    outputs=(MeshOutput(),),
                ),
                Step(
                    name="solve",
                    simulation=simulation,
                    inputs=(MeshInput(source_step="mesh"),),
                    outputs=(SimulationDataOutput(data_type_name="HeatChargeSimulationData"),),
                ),
            )
        )


# =============================================================================
# Workflow Registry
# =============================================================================

# Maps simulation types to their workflow classes
# This allows automatic workflow construction for known multi-step simulations
SIMULATION_TO_WORKFLOW: dict[type, type[Workflow]] = {}


def _register_workflows() -> None:
    """Register known simulation types with their workflows.

    This is called lazily to avoid import issues.
    """
    global SIMULATION_TO_WORKFLOW

    if SIMULATION_TO_WORKFLOW:
        return  # Already registered

    SIMULATION_TO_WORKFLOW = {
        HeatChargeSimulation: HeatChargeWorkflow,
    }


def get_workflow_for_simulation(simulation: WorkflowType) -> Optional[Workflow]:
    """Get the appropriate workflow for a simulation, if it requires one.

    Parameters
    ----------
    simulation : WorkflowType
        The simulation to check.

    Returns
    -------
    Optional[Workflow]
        A workflow for the simulation, or None if it's a single-step simulation.
    """
    _register_workflows()

    workflow_cls = SIMULATION_TO_WORKFLOW.get(type(simulation))
    if workflow_cls:
        return workflow_cls.from_simulation(simulation)
    return None


def is_multi_step_simulation(simulation: WorkflowType) -> bool:
    """Check if a simulation requires a multi-step workflow.

    Parameters
    ----------
    simulation : WorkflowType
        The simulation to check.

    Returns
    -------
    bool
        True if the simulation requires multiple steps.
    """
    _register_workflows()
    return type(simulation) in SIMULATION_TO_WORKFLOW

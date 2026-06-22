from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator, model_validator

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.components.tcad.mesher import VolumeMesher
from tidy3d.components.tcad.simulation.heat import HeatSimulation
from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation
from tidy3d.components.types.base import discriminated_union
from tidy3d.components.types.workflow import WorkflowOperationType
from tidy3d.exceptions import ValidationError

if TYPE_CHECKING:
    from tidy3d.compat import Self


StepOutputKind = Literal[
    "volume_mesh",
    "SimulationData",
    "HeatChargeSimulationData",
    "HeatSimulationData",
    "EMESimulationData",
    "MicrowaveModeSolverData",
    "ModeSolverData",
    "ModeSimulationData",
    "VolumeMesherData",
    "ModalComponentModelerData",
    "TerminalComponentModelerData",
]
StepOutputUsage = Literal["dependency", "load", "both"]


class StepInput(Tidy3dBaseModel):
    """Reference to a named output produced by a previous step."""

    upstream_step: str = Field(
        title="Upstream Step",
        description="Name of the upstream step this input depends on.",
    )

    upstream_output: str = Field(
        title="Upstream Output",
        description="Name of the upstream step output consumed by this input.",
    )

    @field_validator("upstream_step", "upstream_output")
    @classmethod
    def _validate_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Workflow input fields cannot be empty.")
        return value


class StepOutput(Tidy3dBaseModel):
    """Output produced by a workflow step."""

    kind: StepOutputKind = Field(
        title="Output Kind",
        description="Serializable identifier for the output type or artifact kind.",
    )

    usage: StepOutputUsage = Field(
        title="Output Usage",
        description=(
            "Whether the output is used for workflow dependencies, step loading, or both."
        ),
    )

    optional: bool = Field(
        False,
        title="Optional",
        description="Whether the output may be absent or trivial depending on step configuration.",
    )

    default_load: bool = Field(
        False,
        title="Default Load",
        description="Whether this is the default user-facing output for step-loading semantics.",
    )

    @model_validator(mode="after")
    def _validate_default_load_usage(self) -> Self:
        if self.default_load and self.usage == "dependency":
            self._raise_validation_error_at_loc(
                "Workflow outputs marked as 'default_load' must use usage 'load' or 'both'.",
                "default_load",
            )
        return self


class Step(Tidy3dBaseModel):
    """A single step in a workflow."""

    name: str = Field(
        title="Step Name",
        description="Unique step name within a workflow.",
    )

    operation: discriminated_union(WorkflowOperationType) = Field(
        title="Step Operation",
        description="Operation executed for this step.",
    )

    inputs: tuple[StepInput, ...] = Field(
        (),
        title="Inputs",
        description="Inputs consumed by this step.",
    )

    outputs: dict[str, StepOutput] = Field(
        default_factory=dict,
        title="Outputs",
        description="Outputs produced by this step, keyed by output name.",
    )

    cacheable: bool = Field(
        False,
        title="Cacheable",
        description="Whether this step can be safely restored from local cache.",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Workflow step name cannot be empty.")
        return value

    @model_validator(mode="after")
    def _validate_outputs(self) -> Self:
        for output_name in self.outputs:
            if not output_name.strip():
                self._raise_validation_error_at_loc(
                    "Workflow output names cannot be empty.",
                    "outputs",
                    output_name,
                )
        return self

    def get_output(self, output_name: str) -> StepOutput | None:
        """Return a named output if it exists."""
        return self.outputs.get(output_name)

    @property
    def dependency_outputs(self) -> tuple[StepOutput, ...]:
        """Outputs intended for step-to-step workflow dependencies."""
        return tuple(
            output for output in self.outputs.values() if output.usage in ("dependency", "both")
        )

    @property
    def loadable_outputs(self) -> tuple[StepOutput, ...]:
        """Outputs intentionally exposed through step-loading convenience APIs."""
        return tuple(output for output in self.outputs.values() if output.usage in ("load", "both"))

    @property
    def default_load_output(self) -> StepOutput | None:
        """Default user-facing output for step-loading semantics, when unambiguous."""
        default_outputs = tuple(output for output in self.loadable_outputs if output.default_load)
        if default_outputs:
            return default_outputs[0]
        if len(self.loadable_outputs) == 1:
            return self.loadable_outputs[0]
        return None

    @property
    def default_load_output_name(self) -> str | None:
        """Default user-facing output name for step-loading semantics, when unambiguous."""
        default_output = self.default_load_output
        if default_output is None:
            return None
        for output_name, output in self.outputs.items():
            if output is default_output:
                return output_name
        return None


class Workflow(Tidy3dBaseModel):
    """Ordered sequence of workflow steps."""

    steps: tuple[Step, ...] = Field(
        title="Steps",
        description="Ordered workflow steps.",
    )

    @model_validator(mode="after")
    def _validate_steps(self) -> Self:
        if not self.steps:
            self._raise_validation_error_at_loc(
                ValidationError("Workflow must contain at least one step."), "steps"
            )

        seen_steps: dict[str, Step] = {}
        for step_index, step in enumerate(self.steps):
            if step.name in seen_steps:
                self._raise_validation_error_at_loc(
                    ValidationError(f"Duplicate workflow step name '{step.name}'."),
                    "steps",
                    step_index,
                    "name",
                )

            default_load_count = 0
            default_load_output_name = None
            for output_name, output in step.outputs.items():
                if output.default_load:
                    default_load_count += 1
                    default_load_output_name = output_name

            if default_load_count > 1:
                self._raise_validation_error_at_loc(
                    ValidationError(
                        f"Step '{step.name}' defines multiple outputs marked as 'default_load'."
                    ),
                    "steps",
                    step_index,
                    "outputs",
                    default_load_output_name,
                    "default_load",
                )

            if step.outputs and not step.loadable_outputs:
                self._raise_validation_error_at_loc(
                    ValidationError(
                        f"Step '{step.name}' exposes only workflow dependency outputs and must "
                        "also expose a loadable output for Job.step() and Job.load_step()."
                    ),
                    "steps",
                    step_index,
                    "outputs",
                )

            if len(step.dependency_outputs) > 1:
                self._raise_validation_error_at_loc(
                    ValidationError(
                        f"Step '{step.name}' exposes multiple workflow dependency outputs. "
                        "A workflow step may expose at most one dependency output."
                    ),
                    "steps",
                    step_index,
                    "outputs",
                )

            if len(step.loadable_outputs) > 1:
                self._raise_validation_error_at_loc(
                    ValidationError(
                        f"Step '{step.name}' exposes multiple loadable outputs. "
                        "Job.step() and Job.load_step() require at most one loadable "
                        "output per workflow step."
                    ),
                    "steps",
                    step_index,
                    "outputs",
                )

            for input_index, step_input in enumerate(step.inputs):
                upstream_step = seen_steps.get(step_input.upstream_step)
                if upstream_step is None:
                    self._raise_validation_error_at_loc(
                        ValidationError(
                            f"Step '{step.name}' input references unknown or future step "
                            f"'{step_input.upstream_step}'."
                        ),
                        "steps",
                        step_index,
                        "inputs",
                        input_index,
                        "upstream_step",
                    )
                    continue
                upstream_output = upstream_step.get_output(step_input.upstream_output)
                if upstream_output is None:
                    self._raise_validation_error_at_loc(
                        ValidationError(
                            f"Step '{step.name}' input references unknown output "
                            f"'{step_input.upstream_output}' from step '{step_input.upstream_step}'."
                        ),
                        "steps",
                        step_index,
                        "inputs",
                        input_index,
                        "upstream_output",
                    )
                    continue
                if upstream_output.usage not in ("dependency", "both"):
                    self._raise_validation_error_at_loc(
                        ValidationError(
                            f"Step '{step.name}' input references output "
                            f"'{step_input.upstream_output}' "
                            f"from step '{step_input.upstream_step}', but that output is not "
                            "available as a workflow dependency."
                        ),
                        "steps",
                        step_index,
                        "inputs",
                        input_index,
                        "upstream_output",
                    )
            seen_steps[step.name] = step
        return self


class HeatChargeWorkflow(Workflow):
    """Workflow for HeatCharge simulations: mesh -> solve."""

    @staticmethod
    def _simulation_data_kind(simulation: HeatChargeSimulation) -> StepOutputKind:
        """Return the user-facing solve-step data kind for the simulation class."""
        if isinstance(simulation, HeatSimulation):
            return "HeatSimulationData"
        return "HeatChargeSimulationData"

    @classmethod
    def from_simulation(cls, simulation: HeatChargeSimulation) -> HeatChargeWorkflow:
        mesh_step = Step(
            name="mesh",
            # Mesh-step data is empty by default unless VolumeMeshMonitors are configured.
            # TODO(EMCORE-0003): expose user-defined mesh-monitor configuration and
            # direct meshing-step controls through workflow APIs.
            operation=VolumeMesher(simulation=simulation),
            outputs={
                "volume_mesh": StepOutput(kind="volume_mesh", usage="dependency"),
                "mesh_data": StepOutput(
                    kind="VolumeMesherData",
                    usage="load",
                    default_load=True,
                ),
            },
            cacheable=True,
        )
        solve_step = Step(
            name="solve",
            operation=simulation,
            inputs=(StepInput(upstream_step=mesh_step.name, upstream_output="volume_mesh"),),
            outputs={
                "simulation_data": StepOutput(
                    kind=cls._simulation_data_kind(simulation),
                    usage="load",
                    default_load=True,
                ),
            },
            cacheable=True,
        )
        return cls(steps=(mesh_step, solve_step))


SIMULATION_TO_WORKFLOW: dict[type, type[Workflow]] = {
    HeatSimulation: HeatChargeWorkflow,
    HeatChargeSimulation: HeatChargeWorkflow,
}


def resolve_workflow(
    simulation: WorkflowOperationType, workflow: Workflow | None = None
) -> Workflow:
    """Resolve a workflow for a simulation, defaulting to a single-step workflow."""
    if workflow is not None:
        return workflow

    for simulation_type, workflow_type in SIMULATION_TO_WORKFLOW.items():
        if isinstance(simulation, simulation_type):
            return workflow_type.from_simulation(simulation)

    return Workflow(steps=(Step(name="execute", operation=simulation, cacheable=True),))

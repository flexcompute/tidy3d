"""Web-runtime dependency rules for workflow parent-task inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tidy3d.components.tcad.mesher import VolumeMesher
from tidy3d.components.tcad.simulation.heat import HeatSimulation
from tidy3d.components.tcad.simulation.heat_charge import HeatChargeSimulation

if TYPE_CHECKING:
    from tidy3d.components.workflow import Step, StepOutput, StepOutputKind, Workflow


@dataclass(frozen=True)
class ParentTaskDependencyRule:
    """Runtime rule supported by Tidy3D web execution for a parent-task dependency."""

    name: str
    upstream_operation_type: type
    downstream_operation_type: type
    upstream_output_name: str
    output_kind: StepOutputKind
    single_input_only: bool
    description: str

    def matches(
        self,
        step: Step,
        upstream_step: Step,
        upstream_output_name: str,
        upstream_output: StepOutput,
    ) -> bool:
        """Whether this rule can execute the declared step dependency."""
        return (
            (not self.single_input_only or len(step.inputs) == 1)
            and isinstance(upstream_step.operation, self.upstream_operation_type)
            and isinstance(step.operation, self.downstream_operation_type)
            and upstream_output_name == self.upstream_output_name
            and upstream_output.kind == self.output_kind
            and upstream_output.usage in ("dependency", "both")
        )


PARENT_TASK_DEPENDENCY_RULES: tuple[ParentTaskDependencyRule, ...] = (
    ParentTaskDependencyRule(
        name="heat_volume_mesh",
        upstream_operation_type=VolumeMesher,
        downstream_operation_type=HeatSimulation,
        upstream_output_name="volume_mesh",
        output_kind="volume_mesh",
        single_input_only=True,
        description=(
            "Heat and HeatCharge workflow dependencies must use exactly one input: "
            "the built-in 'volume_mesh' output from a VolumeMesher step."
        ),
    ),
    ParentTaskDependencyRule(
        name="heat_charge_volume_mesh",
        upstream_operation_type=VolumeMesher,
        downstream_operation_type=HeatChargeSimulation,
        upstream_output_name="volume_mesh",
        output_kind="volume_mesh",
        single_input_only=True,
        description=(
            "Heat and HeatCharge workflow dependencies must use exactly one input: "
            "the built-in 'volume_mesh' output from a VolumeMesher step."
        ),
    ),
)


def parent_task_dependency_rule(
    step: Step,
    upstream_step: Step,
    upstream_output_name: str,
    upstream_output: StepOutput,
) -> ParentTaskDependencyRule | None:
    """Return the registered runtime rule for a parent-task dependency, if supported."""
    for rule in PARENT_TASK_DEPENDENCY_RULES:
        if rule.matches(
            step=step,
            upstream_step=upstream_step,
            upstream_output_name=upstream_output_name,
            upstream_output=upstream_output,
        ):
            return rule
    return None


def is_supported_parent_task_input(
    step: Step,
    upstream_step: Step,
    upstream_output_name: str,
    upstream_output: StepOutput,
) -> bool:
    """Whether an input can be routed through Tidy3D web parent-task submission."""
    return (
        parent_task_dependency_rule(step, upstream_step, upstream_output_name, upstream_output)
        is not None
    )


def unsupported_parent_task_dependency_message() -> str:
    """Return a concise validation hint for unsupported parent-task dependencies."""
    supported_descriptions = {
        rule.description for rule in PARENT_TASK_DEPENDENCY_RULES if rule.description
    }
    if supported_descriptions:
        return " ".join(sorted(supported_descriptions))
    return "This workflow dependency is not supported by Tidy3D web execution."


def has_supported_parent_task_dependency(steps: tuple[Step, ...]) -> bool:
    """Whether the workflow contains a dependency supported by parent-task execution."""
    if len(steps) < 2:
        return False

    first_step = steps[0]
    if not hasattr(first_step, "get_output"):
        return False
    for step in steps[1:]:
        for step_input in getattr(step, "inputs", ()):
            if step_input.upstream_step != first_step.name:
                continue
            upstream_output = first_step.get_output(step_input.upstream_output)
            if upstream_output is None:
                continue
            if (
                parent_task_dependency_rule(
                    step,
                    first_step,
                    step_input.upstream_output,
                    upstream_output,
                )
                is not None
            ):
                return True
    return False


def supports_implicit_parent_task_reuse(steps: tuple[Step, ...], workflow: Workflow | None) -> bool:
    """Whether top-level ``parent_tasks`` may satisfy the first workflow step."""
    return workflow is None and has_supported_parent_task_dependency(steps)

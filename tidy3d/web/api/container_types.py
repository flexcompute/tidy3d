"""Shared recursive batch container types."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import TypeAliasType

from tidy3d.components.types.workflow import WorkflowDataType, WorkflowOperationType

if TYPE_CHECKING:
    pass

BatchTaskTree = TypeAliasType(
    "BatchTaskTree",
    str | tuple["BatchTaskTree", ...] | dict[str, "BatchTaskTree"],
)

BatchInput = TypeAliasType(
    "BatchInput",
    WorkflowOperationType | list["BatchInput"] | tuple["BatchInput", ...] | dict[str, "BatchInput"],
)
BatchOutput = TypeAliasType(
    "BatchOutput",
    WorkflowDataType | list["BatchOutput"] | tuple["BatchOutput", ...] | dict[str, "BatchOutput"],
)

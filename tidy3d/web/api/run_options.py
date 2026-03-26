"""Helpers for config-backed web run options."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from tidy3d.config import config
from tidy3d.config.sections import VALID_VGPU_ALLOCATIONS
from tidy3d.log import log
from tidy3d.web.core.types import PayType

if TYPE_CHECKING:
    from typing import Union


@dataclass(frozen=True)
class ResolvedUploadOptions:
    """Resolved upload options after applying config defaults."""

    solver_version: Optional[str]
    simulation_type: str


@dataclass(frozen=True)
class ResolvedRunStartOptions:
    """Resolved config-backed run options for task start."""

    solver_version: Optional[str]
    worker_group: Optional[str]
    additional_payload: Optional[dict[str, Any]]


@dataclass(frozen=True)
class ResolvedVgpuStartOptions:
    """Resolved config-backed vGPU options for task start."""

    priority: Optional[int]
    vgpu_allocation: Optional[int]
    ignore_memory_limit: Optional[bool]


def log_deprecated_run_args(
    *,
    solver_version: Optional[str] = None,
    worker_group: Optional[str] = None,
    simulation_type: Optional[str] = None,
    pay_type: Optional[Union[PayType, str]] = None,
    priority: Optional[int] = None,
    vgpu_allocation: Optional[int] = None,
    ignore_memory_limit: Optional[bool] = None,
) -> None:
    """Log a single deprecation hint for legacy run arguments."""

    if all(
        value is None
        for value in (
            solver_version,
            worker_group,
            simulation_type,
            pay_type,
            priority,
            vgpu_allocation,
            ignore_memory_limit,
        )
    ):
        return

    log.warning(
        "Passing run options as direct arguments is deprecated. "
        "Set defaults via 'td.config.run' and 'td.config.vgpu' instead.",
        log_once=True,
    )


def resolve_upload_options(
    *,
    solver_version: Optional[str],
    simulation_type: Optional[str],
) -> ResolvedUploadOptions:
    """Resolve upload options by applying config defaults."""

    return ResolvedUploadOptions(
        solver_version=solver_version if solver_version is not None else config.run.solver_version,
        simulation_type=(
            simulation_type if simulation_type is not None else config.run.simulation_type
        ),
    )


def _resolve_additional_payload() -> Optional[dict[str, Any]]:
    """Resolve the additional submit payload from config."""

    additional_payload = config.run.additional_payload
    if additional_payload is None:
        return None
    return dict(additional_payload)


def resolve_run_start_options(
    *,
    solver_version: Optional[str],
    worker_group: Optional[str],
) -> ResolvedRunStartOptions:
    """Resolve config-backed run options for task start."""

    return ResolvedRunStartOptions(
        solver_version=solver_version if solver_version is not None else config.run.solver_version,
        worker_group=(worker_group if worker_group is not None else config.run.worker_group),
        additional_payload=_resolve_additional_payload(),
    )


def resolve_pay_type(
    pay_type: Optional[Union[PayType, str]], *, apply_config_default: bool = True
) -> PayType:
    """Resolve a pay type override against config defaults."""

    resolved_pay_type = (
        pay_type if pay_type is not None else config.run.pay_type if apply_config_default else None
    )
    return PayType.AUTO if resolved_pay_type is None else PayType(resolved_pay_type)


def resolve_vgpu_start_options(
    *,
    priority: Optional[int],
    vgpu_allocation: Optional[int],
    ignore_memory_limit: Optional[bool],
    apply_config_defaults: bool = True,
) -> ResolvedVgpuStartOptions:
    """Resolve config-backed vGPU options for task start."""

    resolved_priority = (
        priority
        if priority is not None
        else config.vgpu.priority
        if apply_config_defaults
        else None
    )
    if resolved_priority is not None and (resolved_priority < 1 or resolved_priority > 10):
        raise ValueError("Priority must be between '1' and '10' if specified.")

    resolved_vgpu_allocation = (
        vgpu_allocation
        if vgpu_allocation is not None
        else config.vgpu.vgpu_allocation
        if apply_config_defaults
        else None
    )
    if (
        resolved_vgpu_allocation is not None
        and resolved_vgpu_allocation not in VALID_VGPU_ALLOCATIONS
    ):
        raise ValueError(
            f"vgpu_allocation must be one of {list(VALID_VGPU_ALLOCATIONS)} if specified."
        )

    return ResolvedVgpuStartOptions(
        priority=resolved_priority,
        vgpu_allocation=resolved_vgpu_allocation,
        ignore_memory_limit=(
            ignore_memory_limit
            if ignore_memory_limit is not None
            else config.vgpu.ignore_memory_limit
            if apply_config_defaults
            else None
        ),
    )

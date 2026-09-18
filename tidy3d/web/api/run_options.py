"""Helpers for config-backed web run options."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from threading import Lock
from time import monotonic
from typing import Any

from tidy3d.config import config
from tidy3d.log import log
from tidy3d.web.core.http_util import api_key, http
from tidy3d.web.core.types import PayType

_VGPU_ALLOCATION_LIMIT_TTL = 300
_VGPU_ALLOCATION_LIMIT_UNAVAILABLE_TTL = 30
_vgpu_allocation_limits: dict[tuple[str, str | None], tuple[float, int | None]] = {}
_vgpu_allocation_limit_lock = Lock()


@dataclass(frozen=True)
class ResolvedUploadOptions:
    """Resolved upload options after applying config defaults."""

    solver_version: str | None
    simulation_type: str


@dataclass(frozen=True)
class ResolvedRunStartOptions:
    """Resolved config-backed run options for task start."""

    solver_version: str | None
    worker_group: str | None
    additional_payload: dict[str, Any] | None


@dataclass(frozen=True)
class ResolvedVgpuStartOptions:
    """Resolved config-backed vGPU options for task start."""

    priority: int | None
    vgpu_allocation: int | None
    ignore_memory_limit: bool | None


def log_deprecated_run_args(
    *,
    solver_version: str | None = None,
    worker_group: str | None = None,
    simulation_type: str | None = None,
    pay_type: PayType | str | None = None,
    priority: int | None = None,
    vgpu_allocation: int | None = None,
    ignore_memory_limit: bool | None = None,
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
    solver_version: str | None,
    simulation_type: str | None,
) -> ResolvedUploadOptions:
    """Resolve upload options by applying config defaults."""

    return ResolvedUploadOptions(
        solver_version=solver_version if solver_version is not None else config.run.solver_version,
        simulation_type=(
            simulation_type if simulation_type is not None else config.run.simulation_type
        ),
    )


def _resolve_additional_payload() -> dict[str, Any] | None:
    """Resolve the additional submit payload from config."""

    additional_payload = config.run.additional_payload
    if additional_payload is None:
        return None
    return dict(additional_payload)


def resolve_run_start_options(
    *,
    solver_version: str | None,
    worker_group: str | None,
) -> ResolvedRunStartOptions:
    """Resolve config-backed run options for task start."""

    return ResolvedRunStartOptions(
        solver_version=solver_version if solver_version is not None else config.run.solver_version,
        worker_group=(worker_group if worker_group is not None else config.run.worker_group),
        additional_payload=_resolve_additional_payload(),
    )


def resolve_pay_type(
    pay_type: PayType | str | None, *, apply_config_default: bool = True
) -> PayType:
    """Resolve a pay type override against config defaults."""

    resolved_pay_type = (
        pay_type if pay_type is not None else config.run.pay_type if apply_config_default else None
    )
    return PayType.AUTO if resolved_pay_type is None else PayType(resolved_pay_type)


def _get_vgpu_allocation_limit() -> int | None:
    """Get and cache the account's maximum vGPU allocation, if available."""

    key = api_key()
    cache_key = (
        config.web.api_endpoint,
        hashlib.sha256(key.encode()).hexdigest() if key is not None else None,
    )
    now = monotonic()
    with _vgpu_allocation_limit_lock:
        cached = _vgpu_allocation_limits.get(cache_key)
        if cached is not None:
            expires_at, limit = cached
            if expires_at > now:
                return limit
            _vgpu_allocation_limits.pop(cache_key, None)
        try:
            quota = http.get("tidy3d/resources/reservedGpu/quota")
            limit = quota.get("maxConcurrentGpus") if isinstance(quota, dict) else None
            if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
                limit = None
        except Exception as exc:
            log.debug("Unable to determine the vGPU allocation limit: %s", exc)
            limit = None
        ttl = (
            _VGPU_ALLOCATION_LIMIT_TTL
            if limit is not None
            else _VGPU_ALLOCATION_LIMIT_UNAVAILABLE_TTL
        )
        _vgpu_allocation_limits[cache_key] = (now + ttl, limit)
        return limit


def _validate_vgpu_allocation(value: int | None) -> None:
    """Validate a requested vGPU allocation against its known account limit."""

    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"vgpu_allocation must be an integer; got {value!r}.")
    if value < 1:
        raise ValueError(f"vgpu_allocation={value} must be at least 1.")
    limit = _get_vgpu_allocation_limit()
    if limit is not None and value > limit:
        raise ValueError(f"vgpu_allocation={value} exceeds the {limit} vGPU on your license.")


def validate_vgpu_allocation(
    vgpu_allocation: int | None, *, apply_config_default: bool = True
) -> int | None:
    """Validate a vGPU allocation and optionally apply the configured default."""

    resolved = (
        vgpu_allocation
        if vgpu_allocation is not None
        else config.vgpu.vgpu_allocation
        if apply_config_default
        else None
    )
    _validate_vgpu_allocation(resolved)
    return resolved


def resolve_vgpu_start_options(
    *,
    priority: int | None,
    vgpu_allocation: int | None,
    ignore_memory_limit: bool | None,
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

    resolved_vgpu_allocation = validate_vgpu_allocation(
        vgpu_allocation, apply_config_default=apply_config_defaults
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

"""Process-wide cumulative FlexCredit reservations for cloud task starts."""

from __future__ import annotations

import math
import os
import threading
from decimal import Decimal
from typing import TYPE_CHECKING

from tidy3d.exceptions import (
    FlexCreditLimitExceededError,
    WebError,
    format_chained_exception_message,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

EXECUTION_FLEXCREDIT_LIMIT_ENV = "TIDY3D_EXECUTION_FLEXCREDIT_LIMIT"


def _read_execution_flexcredit_limit() -> float | None:
    """Read and validate the process execution FlexCredit limit from the environment."""
    environment_value = os.getenv(EXECUTION_FLEXCREDIT_LIMIT_ENV)
    if environment_value is None:
        return None
    try:
        limit = float(environment_value)
    except (TypeError, ValueError) as exc:
        raise WebError(
            format_chained_exception_message(
                f"Invalid {EXECUTION_FLEXCREDIT_LIMIT_ENV} value: {environment_value!r}. "
                "Expected a non-negative finite number.",
                exc,
            )
        ) from exc
    if not math.isfinite(limit) or limit < 0:
        raise WebError(
            f"Invalid {EXECUTION_FLEXCREDIT_LIMIT_ENV} value: {environment_value!r}. "
            "Expected a non-negative finite number."
        )
    return limit


def _validated_execution_flexcredit_cost(task_id: str, cost: object) -> float:
    """Validate one server-provided maximum FlexCredit estimate."""
    try:
        validated_cost = float(cost)
    except (TypeError, ValueError) as exc:
        raise WebError(
            format_chained_exception_message(
                "Cannot enforce the Tidy3D execution FlexCredit limit because task "
                f"'{task_id}' returned an invalid maximum cost estimate: {cost!r}.",
                exc,
            )
        ) from exc
    if not math.isfinite(validated_cost) or validated_cost < 0:
        raise WebError(
            "Cannot enforce the Tidy3D execution FlexCredit limit because task "
            f"'{task_id}' returned an invalid maximum cost estimate: {validated_cost!r}."
        )
    return validated_cost


_EXECUTION_FLEXCREDIT_LIMIT = _read_execution_flexcredit_limit()
_EXECUTION_FLEXCREDIT_RESERVATION_LOCK = threading.RLock()
_EXECUTION_FLEXCREDIT_TASK_RESERVATIONS: dict[str, float] = {}


def execution_flexcredit_limit_enabled() -> bool:
    """Return whether a process-wide execution FlexCredit limit is active."""
    return _EXECUTION_FLEXCREDIT_LIMIT is not None


def execution_flexcredit_cost_reserved(task_id: str) -> bool:
    """Return whether a task maximum is already reserved for this Python process."""
    with _EXECUTION_FLEXCREDIT_RESERVATION_LOCK:
        return task_id in _EXECUTION_FLEXCREDIT_TASK_RESERVATIONS


def reserve_execution_flexcredit_costs(task_costs: Mapping[str, float]) -> None:
    """Atomically reserve maximum task estimates before cloud tasks start."""
    if _EXECUTION_FLEXCREDIT_LIMIT is None:
        return

    with _EXECUTION_FLEXCREDIT_RESERVATION_LOCK:
        new_task_costs = {
            task_id: _validated_execution_flexcredit_cost(task_id, cost)
            for task_id, cost in task_costs.items()
            if task_id not in _EXECUTION_FLEXCREDIT_TASK_RESERVATIONS
        }
        if not new_task_costs:
            return

        reserved = sum(
            (Decimal(str(cost)) for cost in _EXECUTION_FLEXCREDIT_TASK_RESERVATIONS.values()),
            start=Decimal(0),
        )
        proposed = sum((Decimal(str(cost)) for cost in new_task_costs.values()), start=Decimal(0))
        proposed_total = reserved + proposed
        limit = Decimal(str(_EXECUTION_FLEXCREDIT_LIMIT))
        if proposed_total > limit:
            raise FlexCreditLimitExceededError(
                limit=float(limit),
                reserved=float(reserved),
                proposed=float(proposed),
                proposed_total=float(proposed_total),
                task_ids=tuple(new_task_costs),
            )

        _EXECUTION_FLEXCREDIT_TASK_RESERVATIONS.update(new_task_costs)


def execution_flexcredit_status() -> dict[str, object]:
    """Return a snapshot of process-wide maximum FlexCredit reservations.

    Returns
    -------
    dict
        A snapshot with ``limit_flexcredits``,
        ``reserved_maximum_flexcredits``, ``remaining_flexcredits``, and
        ``task_reservations``, which maps task IDs to maximum estimates. The limit
        and remaining budget are ``None`` when the execution limit is disabled.
    """
    with _EXECUTION_FLEXCREDIT_RESERVATION_LOCK:
        task_reservations = dict(_EXECUTION_FLEXCREDIT_TASK_RESERVATIONS)
        reserved = sum(
            (Decimal(str(cost)) for cost in task_reservations.values()), start=Decimal(0)
        )
        limit = (
            None
            if _EXECUTION_FLEXCREDIT_LIMIT is None
            else Decimal(str(_EXECUTION_FLEXCREDIT_LIMIT))
        )

        return {
            "limit_flexcredits": None if limit is None else float(limit),
            "reserved_maximum_flexcredits": float(reserved),
            "remaining_flexcredits": None if limit is None else float(limit - reserved),
            "task_reservations": task_reservations,
        }


def _reset_execution_flexcredit_reservations() -> None:
    """Reset cumulative execution FlexCredit reservations for isolated tests."""
    with _EXECUTION_FLEXCREDIT_RESERVATION_LOCK:
        _EXECUTION_FLEXCREDIT_TASK_RESERVATIONS.clear()

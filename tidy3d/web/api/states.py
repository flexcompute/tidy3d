from __future__ import annotations

PRE_ERROR_STATES = {
    "aborting",
}

ERROR_STATES = {
    "validate_error",
    "error",
    "errored",
    "blocked",
    "preprocess_error",
    "run_error",
    "aborted",
    "deleted",
    "postprocess_error",
}

PRE_VALIDATE_STATES = {
    "draft",
    "queued",
    "queued_solver",
    "preprocess",
    "validating",
    "validate",
}

POST_RUN_STATES = {
    "postprocess",
    "run_success",
}

COMPLETED_STATES = {
    "visualize",
    "success",
    "completed",
    "processed",
    "postprocess_success",
    "diverge",
    "diverged",
}

END_STATES = ERROR_STATES | COMPLETED_STATES

POST_VALIDATE_STATES = {"validate_success", "validate_warn"}

RUNNING_STATES = (
    PRE_VALIDATE_STATES | POST_VALIDATE_STATES | {"running"} | POST_RUN_STATES | COMPLETED_STATES
)

ALL_POST_VALIDATE_STATES = POST_VALIDATE_STATES | {"running"} | POST_RUN_STATES | END_STATES

VALID_PROGRESS_STATES = RUNNING_STATES | PRE_ERROR_STATES

ALL_STATES = VALID_PROGRESS_STATES | ERROR_STATES


PROGRESSION_ORDER = (
    "draft",
    "queued",
    "queued_solver",
    "preprocess",
    "validating",
    "validate",
    "validate_success",
    "validate_warn",
    "running",
    "postprocess",
    "run_success",
    "visualize",
    "success",
    "completed",
)

MAX_STEPS = len(PROGRESSION_ORDER) - 1
COMPLETED_PERCENT = 100


STATE_PROGRESS_PERCENTAGE = {
    # --- Progression States ---
    "draft": round((0 / MAX_STEPS) * COMPLETED_PERCENT),  # 0%
    "queued": round((1 / MAX_STEPS) * COMPLETED_PERCENT),  # 8%
    "queued_solver": round((2 / MAX_STEPS) * COMPLETED_PERCENT),  # 15%
    "preprocess": round((3 / MAX_STEPS) * COMPLETED_PERCENT),  # 23%
    "validating": round((4 / MAX_STEPS) * COMPLETED_PERCENT),  # 31%
    "validate": round((5 / MAX_STEPS) * COMPLETED_PERCENT),  # 38%
    "validate_success": round((6 / MAX_STEPS) * COMPLETED_PERCENT),  # 46%
    "validate_warn": round((7 / MAX_STEPS) * COMPLETED_PERCENT),  # 54%
    "running": round((8 / MAX_STEPS) * COMPLETED_PERCENT),  # 62%
    "run_success": round((9 / MAX_STEPS) * COMPLETED_PERCENT),  # 69%
    "postprocess": round((10 / MAX_STEPS) * COMPLETED_PERCENT),  # 77%
    "visualize": round((11 / MAX_STEPS) * COMPLETED_PERCENT),  # 85%
    "success": COMPLETED_PERCENT,  # 100%
    "completed": COMPLETED_PERCENT,  # 100%
    "postprocess_success": COMPLETED_PERCENT,  # 100%
    # --- Error States ---
    # All error states map to 0%
    "validate_fail": 0,
    "error": 0,
    "errored": 0,
    "diverge": 100,
    "diverged": 100,
    "blocked": 0,
    "run_failed": 0,
    "aborted": 0,
    "deleted": 0,
    "validate_error": 0,
    "preprocess_error": 0,
    "run_error": 0,
    "postprocess_error": 0,
}

from __future__ import annotations

# progression order for a typical run
PROGRESSION_ORDER = (
    "draft",
    "queued",
    "preprocess",
    "running",
    "postprocess",
    "success",
)

MAX_STEPS = len(PROGRESSION_ORDER) - 1
COMPLETED_PERCENT = 100

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
    "validating",
    "validate",
}

QUEUED_STATES = {"queued", "queued_solver"}

PREPROCESS_STATES = {"preprocess"}

RUNNING_STATES = {"running", "preprocess_success"}

POSTPROCESS_STATES = {
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

POST_VALIDATE_STATES = {"validate_success", "validate_warn", "warning"}

ALL_POST_VALIDATE_STATES = POST_VALIDATE_STATES | RUNNING_STATES | POSTPROCESS_STATES | END_STATES

VALID_PROGRESS_STATES = (
    PRE_VALIDATE_STATES
    | QUEUED_STATES
    | POST_VALIDATE_STATES
    | RUNNING_STATES
    | POSTPROCESS_STATES
    | COMPLETED_STATES
    | PRE_ERROR_STATES
)

ALL_STATES = VALID_PROGRESS_STATES | ERROR_STATES

STATE_PROGRESS_PERCENTAGE = dict.fromkeys(ALL_STATES, 0)
STATE_PROGRESS_PERCENTAGE.update(dict.fromkeys(COMPLETED_STATES, COMPLETED_PERCENT))
STATE_PROGRESS_PERCENTAGE.update(
    {state: round((1 / MAX_STEPS) * COMPLETED_PERCENT) for state in QUEUED_STATES}
)
STATE_PROGRESS_PERCENTAGE.update(
    {state: round((2 / MAX_STEPS) * COMPLETED_PERCENT) for state in PREPROCESS_STATES}
)
STATE_PROGRESS_PERCENTAGE.update(
    {state: round((3 / MAX_STEPS) * COMPLETED_PERCENT) for state in RUNNING_STATES}
)
STATE_PROGRESS_PERCENTAGE.update(
    {state: round((4 / MAX_STEPS) * COMPLETED_PERCENT) for state in POSTPROCESS_STATES}
)

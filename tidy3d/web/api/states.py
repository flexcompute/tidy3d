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

MAX_STEPS = 4
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

SUCCESS_STATES = {
    "visualize",
    "success",
    "completed",
    "processed",
    "postprocess_success",
}

DIVERGED_STATES = {
    "diverge",
    "diverged",
}

COMPLETED_STATES = DIVERGED_STATES | SUCCESS_STATES

END_STATES = ERROR_STATES | COMPLETED_STATES

POST_VALIDATE_STATES = {"validate_success", "validate_warn", "warning"}

DRAFT_STATES = PRE_VALIDATE_STATES | POST_VALIDATE_STATES

ALL_POST_VALIDATE_STATES = POST_VALIDATE_STATES | RUNNING_STATES | POSTPROCESS_STATES | END_STATES

VALID_PROGRESS_STATES = (
    DRAFT_STATES
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
    {state: round((0 / MAX_STEPS) * COMPLETED_PERCENT) for state in QUEUED_STATES}
)
STATE_PROGRESS_PERCENTAGE.update(
    {state: round((0 / MAX_STEPS) * COMPLETED_PERCENT) for state in DIVERGED_STATES}
)
STATE_PROGRESS_PERCENTAGE.update(
    {state: round((1 / MAX_STEPS) * COMPLETED_PERCENT) for state in PREPROCESS_STATES}
)
STATE_PROGRESS_PERCENTAGE.update(
    {state: round((2 / MAX_STEPS) * COMPLETED_PERCENT) for state in RUNNING_STATES}
)
STATE_PROGRESS_PERCENTAGE.update(
    {state: round((3 / MAX_STEPS) * COMPLETED_PERCENT) for state in POSTPROCESS_STATES}
)
STATE_PROGRESS_PERCENTAGE.update(
    {state: round((4 / MAX_STEPS) * COMPLETED_PERCENT) for state in SUCCESS_STATES}
)


def status_to_stage(status: str) -> tuple[str, int]:
    """Map task status to monotonic stage for progress bars.

    Parameters
    ----------
    status : str
        The task status string.

    Returns
    -------
    tuple[str, int]
        A tuple of (stage_name, stage_index) where stage_index corresponds
        to the position in PROGRESSION_ORDER.
    """
    s = (status or "").lower()
    if s in DRAFT_STATES:
        return ("draft", 0)
    if s in QUEUED_STATES:
        return ("queued", 0)
    if s in PREPROCESS_STATES:
        return ("preprocess", 1)
    if s in RUNNING_STATES:
        return ("running", 2)
    if s in POSTPROCESS_STATES:
        return ("postprocess", 3)
    if s in COMPLETED_STATES:
        return ("success", 4)
    # Unknown states map to earliest stage to avoid showing 100% prematurely
    return (s or "unknown", 0)

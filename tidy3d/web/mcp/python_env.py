"""Python environment detection through the local Tidy3D extension bridge."""

from __future__ import annotations

from typing import Any

from ._dispatcher import invoke_extension_route

DetectionPayload = dict[str, Any]


class PythonEnvironmentDetectionError(RuntimeError):
    """Raised when the extension bridge cannot resolve a Python environment."""


def detect_python_environment_payload(resource: str | None = None) -> DetectionPayload:
    """Resolve the active Python interpreter, environment manager, and project manager."""

    params: dict[str, object | None] = {}
    if resource:
        params["resource"] = resource
    result = invoke_extension_route("/python/detect", params, timeout=10.0)
    error_msg = result.get("error")
    if isinstance(error_msg, str) and error_msg:
        raise PythonEnvironmentDetectionError(f"python detection failed: {error_msg}")
    payload = result.get("result")
    return payload if isinstance(payload, dict) else {"detectionSource": "none"}


def python_environment_summary(payload: DetectionPayload) -> str:
    """Build a short text summary for an environment detection payload."""

    python_exec = payload.get("pythonExec")
    env_manager = payload.get("envManager")
    project_manager = payload.get("projectManager")
    summary_bits = []
    if python_exec:
        summary_bits.append(f"interpreter: {python_exec}")
    if env_manager:
        summary_bits.append(f"env manager: {env_manager}")
    if project_manager:
        summary_bits.append(f"project manager: {project_manager}")
    detection_source = payload.get("detectionSource")
    summary = "; ".join(summary_bits) or "No Python environment detected"
    if detection_source:
        summary = f"{summary} (source={detection_source})"
    return summary

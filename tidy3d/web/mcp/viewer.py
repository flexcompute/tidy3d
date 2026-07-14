"""Viewer bridge operations for the Tidy3D MCP server."""

from __future__ import annotations

import base64
import hashlib
import json
import nturl2path
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypedDict
from urllib.parse import unquote, urlparse

from ._dispatcher import invoke_viewer_command


class ViewerToolError(RuntimeError):
    """Raised when the local viewer bridge rejects a viewer operation."""


_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[a-zA-Z]:[\\/]")


class SlicePayload(TypedDict, total=False):
    """Code slice returned by the viewer bridge."""

    code: str
    requirements: list[str]


class ValidationPayload(TypedDict, total=False):
    """Structured validate_simulation response."""

    viewer_id: str
    status: str
    error: str
    window_id: str
    warnings: list[str]
    slice: SlicePayload


class RotatePayload(TypedDict):
    """Structured rotate_viewer response."""

    viewer_id: str
    direction: str
    status: str


class VisibilityPayload(TypedDict, total=False):
    """Structured show_structures response."""

    viewer_id: str
    status: str
    visibility: list[bool]


def normalize_visibility(entry: object) -> bool:
    """Normalize bridge visibility inputs into booleans."""

    if isinstance(entry, bool):
        return entry
    if entry is None:
        return False
    if isinstance(entry, (int, float)):
        return entry != 0
    if isinstance(entry, str):
        value = entry.strip().lower()
        if value in {"true", "1", "yes", "on"}:
            return True
        if value in {"false", "0", "no", "off", ""}:
            return False
    return bool(entry)


def normalize_warnings(raw: object) -> list[str] | None:
    """Normalize bridge warning payloads into a list of strings."""

    if not raw:
        return None
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Iterable):
        normalized = [str(item) for item in raw if item]
        return normalized or None
    return [str(raw)]


def _local_file_path_text(file: str, *, os_name: str | None = None) -> str | None:
    """Return local filesystem path text, or ``None`` when the input is a remote URI."""

    if _WINDOWS_ABSOLUTE_PATH.match(file) or file.startswith("\\\\"):
        return file
    parsed = urlparse(file)
    if parsed.scheme and parsed.scheme != "file":
        return None
    if parsed.scheme != "file":
        return file
    platform = os.name if os_name is None else os_name
    netloc = "" if parsed.netloc in {"", "localhost"} else parsed.netloc
    if platform == "nt":
        url_path = f"//{netloc}{parsed.path}" if netloc else parsed.path
        return nturl2path.url2pathname(url_path)  # pyrefly: ignore[deprecated]
    path = unquote(parsed.path)
    if netloc:
        return f"//{netloc}{path}"
    return path


def rotate_viewer_payload(viewer_id: str, direction: str) -> RotatePayload:
    """Align the viewer camera to the requested orientation."""

    if not viewer_id:
        raise ValueError("viewer_id is required")
    if not direction:
        raise ValueError("direction is required")
    normalized = direction.upper()
    allowed = {"TOP", "BOTTOM", "LEFT", "RIGHT", "FRONT", "BACK"}
    if normalized not in allowed:
        raise ValueError(f"direction must be one of {sorted(allowed)}")
    result = invoke_viewer_command(
        "rotate", {"viewer": viewer_id, "direction": normalized}, timeout=10.0
    )
    error_msg = result.get("error")
    if isinstance(error_msg, str) and error_msg:
        raise ViewerToolError(f"rotation failed: {error_msg}")
    return {
        "viewer_id": viewer_id,
        "direction": normalized,
        "status": str(result.get("status") or "ok"),
    }


def show_structures_payload(viewer_id: str, visibility: list[object]) -> VisibilityPayload:
    """Toggle structure visibility with normalized boolean flags."""

    if not viewer_id:
        raise ValueError("viewer_id is required")
    flags = [normalize_visibility(entry) for entry in visibility]
    result = invoke_viewer_command(
        "visibility",
        {"viewer": viewer_id, "visibility": json.dumps(flags)},
        timeout=10.0,
    )
    error_msg = result.get("error")
    if isinstance(error_msg, str) and error_msg:
        raise ViewerToolError(f"visibility update failed: {error_msg}")
    response: VisibilityPayload = {
        "viewer_id": viewer_id,
        "status": str(result.get("status") or "ok"),
    }
    returned_flags = result.get("visibility")
    if isinstance(returned_flags, list):
        response["visibility"] = [normalize_visibility(entry) for entry in returned_flags]
    return response


def build_inline_payload(file: str) -> dict[str, str]:
    """Read file content and prepare inline viewer payload."""

    path_text = _local_file_path_text(file)
    if path_text is None:
        return {"source_uri": file}
    candidate = Path(path_text).expanduser()
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ViewerToolError(f"viewer source not found: {candidate}: {exc}") from exc
    except OSError as exc:
        raise ViewerToolError(f"could not resolve viewer source: {candidate}: {exc}") from exc
    try:
        data = resolved.read_bytes()
    except FileNotFoundError as exc:
        raise ViewerToolError(f"viewer source not found: {resolved}: {exc}") from exc
    except OSError as exc:
        raise ViewerToolError(f"could not read viewer source: {resolved}: {exc}") from exc
    encoded = base64.b64encode(data).decode("ascii")
    payload: dict[str, str] = {
        "inline_content": encoded,
        "inline_encoding": "base64",
        "inline_name": resolved.name,
        "inline_token": hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16],
    }
    try:
        payload["source_uri"] = resolved.as_uri()
    except ValueError:
        pass
    try:
        stat = resolved.stat()
    except OSError:
        return payload
    payload["inline_mtime"] = str(stat.st_mtime_ns)
    return payload


def slice_payload(raw: object) -> SlicePayload | None:
    """Normalize a viewer bridge code slice."""

    if not isinstance(raw, dict):
        return None
    code = raw.get("code")
    requirements = raw.get("requirements")
    if not isinstance(code, str) or not code:
        return None
    normalized_requirements = requirements if isinstance(requirements, list) else []
    return {"code": code, "requirements": [str(entry) for entry in normalized_requirements]}


def validate_simulation_payload(
    *,
    file: str | None = None,
    symbol: str | None = None,
    index: int | None = None,
    viewer_id: str | None = None,
) -> ValidationPayload:
    """Validate a simulation file or refresh an existing viewer."""

    launched = False
    start_result: dict[str, Any] | None = None
    start_warnings: list[str] | None = None
    normalized_viewer = viewer_id.strip() if isinstance(viewer_id, str) else None
    inline_payload = build_inline_payload(file) if file else {}

    if not normalized_viewer:
        if not file:
            raise ValueError("file is required when viewer_id is not provided")
        params: dict[str, object | None] = dict(inline_payload)
        if symbol:
            params["symbol"] = symbol
        if index is not None:
            params["index"] = index
        start_result = invoke_viewer_command("start", params, timeout=10.0)
        error_msg = start_result.get("error") if isinstance(start_result, dict) else None
        if isinstance(error_msg, str) and error_msg:
            raise ViewerToolError(f"viewer reported error: {error_msg}")
        resolved = start_result.get("viewer_id") if isinstance(start_result, dict) else None
        if not isinstance(resolved, str) or not resolved:
            raise ValueError("viewer did not confirm readiness")
        normalized_viewer = resolved
        start_warnings = (
            normalize_warnings(start_result.get("warnings") or start_result.get("warning"))
            if isinstance(start_result, dict)
            else None
        )
        launched = True

    params = dict(inline_payload)
    params["viewer"] = normalized_viewer
    check_result = invoke_viewer_command("check", params, timeout=10.0)
    if not isinstance(check_result, dict):
        raise RuntimeError("viewer returned unsupported payload")

    response: ValidationPayload = {"viewer_id": normalized_viewer}
    status = check_result.get("status")
    if not isinstance(status, str) or not status:
        status = start_result.get("status") if isinstance(start_result, dict) else None
    if isinstance(status, str) and status:
        response["status"] = status
    error_msg = check_result.get("error")
    if isinstance(error_msg, str) and error_msg:
        response["error"] = error_msg
    window_id = check_result.get("window_id")
    if isinstance(window_id, str) and window_id:
        response["window_id"] = window_id

    warnings = normalize_warnings(check_result.get("warnings") or check_result.get("warning")) or []
    if start_warnings:
        seen = set(warnings)
        for item in start_warnings:
            if item not in seen:
                warnings.append(item)
                seen.add(item)
    if warnings:
        response["warnings"] = warnings

    check_slice = slice_payload(check_result.get("slice"))
    if check_slice:
        response["slice"] = check_slice
    if launched and "slice" not in response and isinstance(start_result, dict):
        start_slice = slice_payload(start_result.get("slice"))
        if start_slice:
            response["slice"] = start_slice

    return response

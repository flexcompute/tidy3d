"""Viewer screenshot bridge operations for the Tidy3D MCP server."""

from __future__ import annotations

from base64 import b64decode
from typing import TypedDict

from ._dispatcher import invoke_viewer_command


class ScreenshotToolError(RuntimeError):
    """Raised when the viewer cannot provide a screenshot frame."""


class DecodedImage(TypedDict):
    """Decoded image bytes returned by the viewer bridge."""

    data: bytes
    format: str | None
    mime: str | None


class CapturedFrame(TypedDict):
    """Structured capture response before MCP image adaptation."""

    viewer_id: str
    image: DecodedImage


def image_from_data_url(data_url: str) -> DecodedImage:
    """Decode a viewer data URL into bytes and MIME metadata."""

    if not isinstance(data_url, str) or not data_url.startswith("data:"):
        raise ValueError("invalid data URL")
    header, _, payload = data_url.partition(",")
    if not payload:
        raise ValueError("invalid data URL")
    meta = header[5:]
    mime = meta.split(";", 1)[0] if ";" in meta else meta
    try:
        raw = b64decode(payload, validate=True)
    except ValueError as exc:
        raise ValueError(f"invalid data URL: {exc}") from exc
    format_hint = None
    if mime and "/" in mime:
        subtype = mime.split("/", 1)[1]
        if subtype:
            format_hint = subtype.split("+", 1)[0]
    return {"data": raw, "format": format_hint, "mime": mime or None}


def capture_frame_payload(viewer_id: str) -> CapturedFrame:
    """Capture a single frame from the viewer bridge."""

    if not viewer_id:
        raise ValueError("viewer_id is required")
    result = invoke_viewer_command("capture", {"viewer": viewer_id}, timeout=10.0)
    error_msg = result.get("error")
    if isinstance(error_msg, str) and error_msg:
        raise ScreenshotToolError(f"viewer reported error: {error_msg}")
    data_url = result.get("data_url")
    if not isinstance(data_url, str) or not data_url:
        raise ScreenshotToolError("viewer did not return capture data")
    try:
        image = image_from_data_url(data_url)
    except ValueError as exc:
        raise ScreenshotToolError(f"viewer returned invalid image data: {exc}") from exc
    return {"viewer_id": viewer_id, "image": image}

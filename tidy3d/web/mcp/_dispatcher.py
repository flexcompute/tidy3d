"""Viewer bridge dispatch helpers for the Tidy3D MCP server."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from collections.abc import Mapping

_BRIDGE_URL: str | None = None
_LOCAL_BRIDGE_HOSTS = {"127.0.0.1", "localhost", "::1"}


class BridgeConfigurationError(RuntimeError):
    """Raised when a configured bridge endpoint cannot target the local extension."""


def _validated_port(raw: str | int | None) -> int:
    try:
        port = int(raw) if raw is not None else 0
    except (TypeError, ValueError) as exc:
        raise BridgeConfigurationError(
            f"viewer bridge URL must include a valid port; got {raw!r}: {exc}"
        ) from exc
    if port < 1 or port > 65535:
        raise BridgeConfigurationError(
            f"viewer bridge URL port must be between 1 and 65535; got {raw!r}"
        )
    return port


def normalize_bridge_url(candidate: str | None) -> str | None:
    """Normalize a bridge URL or deeplink into a loopback HTTP endpoint.

    Viewer tools can inline local simulation files before dispatching to the
    extension bridge, so configured bridge values must resolve to localhost.
    Invalid non-empty values fail during setup instead of becoming outbound
    requests to arbitrary hosts.
    """

    if not candidate:
        return None
    source = candidate.strip()
    if not source:
        return None
    if "://" not in source:
        if source.startswith(":"):
            source = f"http://127.0.0.1{source}"
        elif source.isdigit():
            source = f"http://127.0.0.1:{source}"
        else:
            source = f"http://{source}"
    parsed = urlparse(source)
    scheme = parsed.scheme or "http"
    if scheme not in {"http", "https"}:
        raise BridgeConfigurationError("viewer bridge URL must use http or https")
    try:
        port = parsed.port
    except ValueError:
        raise BridgeConfigurationError("viewer bridge URL must include a valid port") from None
    port = _validated_port(port)
    hostname = (parsed.hostname or "").lower()
    if hostname not in _LOCAL_BRIDGE_HOSTS:
        raise BridgeConfigurationError("viewer bridge URL must use localhost, 127.0.0.1, or ::1")
    netloc_host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{netloc_host}:{port}"
    path = parsed.path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", parsed.query, "")).rstrip("/")


def configure_dispatcher(bridge_url: str | None) -> str | None:
    """Set the process-wide bridge endpoint discovered during MCP startup."""

    global _BRIDGE_URL
    _BRIDGE_URL = normalize_bridge_url(bridge_url)
    return _BRIDGE_URL


def _bridge_endpoint() -> str | None:
    global _BRIDGE_URL
    if _BRIDGE_URL:
        return _BRIDGE_URL
    _BRIDGE_URL = normalize_bridge_url(os.environ.get("TIDY3D_VIEWER_BRIDGE_URL", ""))
    return _BRIDGE_URL


def _stringify_params(params: Mapping[str, object | None]) -> dict[str, str]:
    return {key: str(value) for key, value in params.items() if value is not None}


def _invoke_bridge_path(
    path: str, params: Mapping[str, object | None], timeout: float
) -> dict[str, Any]:
    endpoint = _bridge_endpoint()
    if not endpoint:
        raise RuntimeError("viewer bridge unavailable; ensure the Tidy3D extension is active")
    payload = _stringify_params(params)
    if timeout and timeout > 0:
        payload["timeout_ms"] = str(int(timeout * 1000))
    data = json.dumps(payload).encode("utf-8")
    parsed = urlparse(endpoint)
    normalized_path = path if path.startswith("/") else f"/{path}"
    full_path = parsed.path.rstrip("/") + normalized_path
    url = urlunparse((parsed.scheme, parsed.netloc, full_path, "", parsed.query, ""))
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"viewer bridge request failed: {exc}") from exc
    text = raw.decode("utf-8") if raw else "{}"
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"bridge returned invalid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise RuntimeError("bridge returned unsupported payload")
    return decoded


def invoke_viewer_command(
    action: str,
    params: Mapping[str, object | None],
    *,
    timeout: float,
) -> dict[str, Any]:
    """Dispatch a viewer command through the local bridge."""

    return _invoke_bridge_path(f"/viewer/{action}", params, timeout)


def invoke_extension_route(
    path: str, params: Mapping[str, object | None], *, timeout: float
) -> dict[str, Any]:
    """Dispatch a non-viewer extension command through the local bridge."""

    return _invoke_bridge_path(path, params, timeout)

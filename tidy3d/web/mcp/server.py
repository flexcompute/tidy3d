"""FastMCP proxy server for Tidy3D."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from tidy3d.web.core.http_util import api_key as configured_api_key

from ._dispatcher import BridgeConfigurationError, configure_dispatcher, normalize_bridge_url

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_REMOTE_MCP_URL = "https://flexagent.simulation.cloud/"
REMOTE_MCP_URL_ENV = "TIDY3D_MCP_REMOTE_URL"
REMOTE_MCP_URL_SETTING = "tidy3d.mcp.remoteUrl"
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; tidy3d-mcp)"


class McpProxyServer(Protocol):
    """Small runtime surface used after FastMCP constructs the proxy."""

    def add_tool(self, tool: object) -> None:
        """Register a local tool with the proxy."""

    def run(self, *, show_banner: bool) -> None:
        """Run the proxy server."""


class ToolFactory(Protocol):
    """FastMCP tool factory surface used for local adapters."""

    def from_function(self, function: Callable[..., object]) -> object:
        """Create a FastMCP tool from a Python callable."""


@dataclass(frozen=True)
class RemoteProxy:
    """FastMCP proxy plus the matching tool factory."""

    proxy: McpProxyServer
    tool_factory: ToolFactory


def _create_remote_proxy(*, api_key: str, mcp_url: str, user_agent: str) -> RemoteProxy:
    """Create the FastMCP proxy, importing FastMCP only on the server path."""

    from fastmcp.client.auth import BearerAuth
    from fastmcp.client.transports import StreamableHttpTransport
    from fastmcp.server import create_proxy
    from fastmcp.tools import Tool

    transport = StreamableHttpTransport(
        mcp_url, headers={"User-Agent": user_agent}, auth=BearerAuth(token=api_key)
    )
    return RemoteProxy(
        proxy=create_proxy(transport, name="Tidy3D"),
        tool_factory=Tool,
    )


def _explicit_bridge_url(explicit: str | None) -> str | None:
    if explicit is None:
        return None
    candidate = explicit.strip()
    if not candidate:
        raise RuntimeError("--viewer-bridge cannot be empty")
    try:
        return normalize_bridge_url(candidate)
    except BridgeConfigurationError as exc:
        raise RuntimeError(f"Invalid --viewer-bridge value: {explicit}: {exc}") from exc


def _environment_bridge_url() -> str | None:
    candidate = os.getenv("TIDY3D_VIEWER_BRIDGE_URL", "").strip()
    if not candidate:
        return None
    try:
        return normalize_bridge_url(candidate)
    except BridgeConfigurationError as exc:
        raise RuntimeError(f"Invalid TIDY3D_VIEWER_BRIDGE_URL value: {candidate}: {exc}") from exc


def _resolve_requested_bridge(explicit: str | None) -> str | None:
    explicit_bridge = _explicit_bridge_url(explicit)
    if explicit_bridge:
        return explicit_bridge
    return _environment_bridge_url()


def _resolve_api_key() -> str:
    """Resolve the API key through the normal Tidy3D credential configuration."""

    candidate = configured_api_key()
    if candidate:
        return candidate
    raise RuntimeError("API key required. Set SIMCLOUD_APIKEY or run `tidy3d configure`.")


def _normalize_remote_mcp_url(mcp_url: str) -> str:
    """Validate that FastMCP receives the exact configured endpoint URL."""
    from urllib.parse import urlsplit

    candidate = mcp_url.strip()
    if not candidate:
        raise RuntimeError(f"{REMOTE_MCP_URL_SETTING} / {REMOTE_MCP_URL_ENV} cannot be empty")

    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(
            f"Invalid {REMOTE_MCP_URL_SETTING} / {REMOTE_MCP_URL_ENV} value: {candidate}: "
            "expected an absolute HTTP(S) URL including the exact MCP endpoint path, "
            "for example `https://host.example/mcp`, or a root-mounted endpoint such as "
            "`https://host.example/`."
        )
    return candidate


def run_mcp_server(
    *,
    viewer_bridge: str | None = None,
) -> None:
    """Launch the Tidy3D FastMCP proxy."""

    resolved_api_key = _resolve_api_key()
    configure_dispatcher(_resolve_requested_bridge(viewer_bridge))

    mcp_url = _normalize_remote_mcp_url(os.getenv(REMOTE_MCP_URL_ENV, DEFAULT_REMOTE_MCP_URL))
    user_agent = os.getenv("TIDY3D_MCP_USER_AGENT", "").strip() or DEFAULT_USER_AGENT
    remote = _create_remote_proxy(
        api_key=resolved_api_key,
        mcp_url=mcp_url,
        user_agent=user_agent,
    )
    from . import tools

    remote.proxy.add_tool(remote.tool_factory.from_function(tools.detect_python_environment))
    remote.proxy.add_tool(remote.tool_factory.from_function(tools.validate_simulation))
    remote.proxy.add_tool(remote.tool_factory.from_function(tools.rotate_viewer))
    remote.proxy.add_tool(remote.tool_factory.from_function(tools.capture))
    remote.proxy.add_tool(remote.tool_factory.from_function(tools.show_structures))
    remote.proxy.run(show_banner=False)


def main(argv: list[str] | None = None) -> None:
    """Module entry point for clients that launch the server with ``python -m``."""

    parser = argparse.ArgumentParser(prog="python -m tidy3d.web.mcp.server")
    parser.add_argument("--viewer-bridge", default=None)
    args = parser.parse_args(argv)
    try:
        run_mcp_server(viewer_bridge=args.viewer_bridge)
    except RuntimeError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

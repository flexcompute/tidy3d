"""Tests for bundled Tidy3D MCP server helpers."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path

import pytest

from tidy3d.web.mcp import _dispatcher, python_env, screenshots, viewer
from tidy3d.web.mcp import server as mcp_server


def test_dispatcher_normalizes_bridge_env_var(monkeypatch):
    monkeypatch.setattr(_dispatcher, "_BRIDGE_URL", None)
    monkeypatch.setenv("TIDY3D_VIEWER_BRIDGE_URL", ":5123")

    assert _dispatcher._bridge_endpoint() == "http://127.0.0.1:5123"


@pytest.mark.parametrize(
    "bridge_url",
    [
        "https://example.com:5123",
        "http://localhost",
        "cursor://Flexcompute.tidy3d/bridge?port=5123",
    ],
)
def test_dispatcher_rejects_non_local_or_incomplete_bridge_url(bridge_url):
    with pytest.raises(RuntimeError):
        _dispatcher.normalize_bridge_url(bridge_url)


def test_viewer_inline_payload_accepts_file_uri(tmp_path):
    source = tmp_path / "simulation.py"
    source.write_bytes(b"sim = object()\n")

    payload = viewer.build_inline_payload(source.as_uri())

    assert payload["inline_name"] == "simulation.py"
    assert payload["source_uri"] == source.as_uri()
    assert base64.b64decode(payload["inline_content"]).decode("utf-8") == "sim = object()\n"


def test_viewer_file_path_text_preserves_windows_absolute_paths():
    assert (
        viewer._local_file_path_text(r"C:\Users\Ada\simulation.py") == r"C:\Users\Ada\simulation.py"
    )
    assert (
        viewer._local_file_path_text(r"\\server\share\simulation.py")
        == r"\\server\share\simulation.py"
    )


def test_viewer_file_uri_uses_windows_path_conversion():
    assert (
        viewer._local_file_path_text("file:///C:/Users/Ada/simulation.py", os_name="nt")
        == r"C:\Users\Ada\simulation.py"
    )
    assert (
        viewer._local_file_path_text("file://localhost/C:/Users/Ada/simulation.py", os_name="nt")
        == r"C:\Users\Ada\simulation.py"
    )
    assert (
        viewer._local_file_path_text("file://server/share/simulation.py", os_name="nt")
        == r"\\server\share\simulation.py"
    )


def test_viewer_inline_payload_passes_workspace_uri_to_bridge():
    assert viewer.build_inline_payload("vscode://file/workspace/simulation.py") == {
        "source_uri": "vscode://file/workspace/simulation.py"
    }


def test_viewer_inline_payload_wraps_read_errors(tmp_path, monkeypatch):
    source = tmp_path / "simulation.py"
    source.write_text("sim = object()\n", encoding="utf-8")

    def fake_read_bytes(_self):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_bytes", fake_read_bytes)

    with pytest.raises(viewer.ViewerToolError, match="could not read viewer source"):
        viewer.build_inline_payload(source.as_uri())


def test_validate_simulation_payload_merges_warnings_and_slice(monkeypatch):
    calls = []

    def fake_invoke(action, params, *, timeout):
        calls.append((action, params, timeout))
        if action == "start":
            return {"viewer_id": "viewer-1", "status": "focused", "warnings": ["start warning"]}
        if action == "check":
            return {
                "status": "ok",
                "warnings": ["check warning"],
                "slice": {"code": "sim = object()", "requirements": ["tidy3d"]},
            }
        pytest.fail(f"unexpected viewer action: {action}")

    monkeypatch.setattr(viewer, "invoke_viewer_command", fake_invoke)

    payload = viewer.validate_simulation_payload(file="vscode://file/workspace/simulation.py")

    assert payload == {
        "viewer_id": "viewer-1",
        "status": "ok",
        "warnings": ["check warning", "start warning"],
        "slice": {"code": "sim = object()", "requirements": ["tidy3d"]},
    }
    assert calls[0] == (
        "start",
        {"source_uri": "vscode://file/workspace/simulation.py"},
        10.0,
    )
    assert calls[1] == (
        "check",
        {"source_uri": "vscode://file/workspace/simulation.py", "viewer": "viewer-1"},
        10.0,
    )


def test_show_structures_payload_normalizes_visibility(monkeypatch):
    captured = {}

    def fake_invoke(action, params, *, timeout):
        captured.update({"action": action, "params": params, "timeout": timeout})
        return {"status": "ok", "visibility": ["true", 0, 1]}

    monkeypatch.setattr(viewer, "invoke_viewer_command", fake_invoke)

    payload = viewer.show_structures_payload("viewer-1", ["yes", "off", 2])

    assert captured == {
        "action": "visibility",
        "params": {"viewer": "viewer-1", "visibility": json.dumps([True, False, True])},
        "timeout": 10.0,
    }
    assert payload == {
        "viewer_id": "viewer-1",
        "status": "ok",
        "visibility": [True, False, True],
    }


def test_capture_frame_payload_decodes_image(monkeypatch):
    png = b"\x89PNG\r\n"
    data_url = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
    monkeypatch.setattr(
        screenshots,
        "invoke_viewer_command",
        lambda *_args, **_kwargs: {"data_url": data_url},
    )

    frame = screenshots.capture_frame_payload("viewer-1")

    assert frame == {
        "viewer_id": "viewer-1",
        "image": {"data": png, "format": "png", "mime": "image/png"},
    }


def test_detect_python_environment_payload_summarizes_bridge_result(monkeypatch):
    monkeypatch.setattr(
        python_env,
        "invoke_extension_route",
        lambda *_args, **_kwargs: {
            "result": {
                "pythonExec": "/env/bin/python",
                "envManager": "venv",
                "projectManager": "uv",
                "detectionSource": "workspace",
            }
        },
    )

    payload = python_env.detect_python_environment_payload()

    assert payload["pythonExec"] == "/env/bin/python"
    assert (
        python_env.python_environment_summary(payload)
        == "interpreter: /env/bin/python; env manager: venv; project manager: uv "
        "(source=workspace)"
    )


def test_fastmcp_adapter_wraps_viewer_payload(monkeypatch):
    from tidy3d.web.mcp import tools

    monkeypatch.setattr(
        viewer,
        "rotate_viewer_payload",
        lambda viewer_id, direction: {
            "viewer_id": viewer_id,
            "direction": direction.upper(),
            "status": "ok",
        },
    )

    result = asyncio.run(tools.rotate_viewer("viewer-1", "top"))

    assert result.structured_content == {
        "viewer_id": "viewer-1",
        "direction": "TOP",
        "status": "ok",
    }
    assert result.content[0].text == "Viewer aligned to TOP"


def test_fastmcp_adapter_wraps_expected_helper_errors(monkeypatch):
    from fastmcp.exceptions import ToolError

    from tidy3d.web.mcp import tools

    def fake_validate(**_kwargs):
        raise ValueError("file is required")

    monkeypatch.setattr(viewer, "validate_simulation_payload", fake_validate)

    with pytest.raises(ToolError, match="file is required"):
        asyncio.run(tools.validate_simulation())


def test_run_mcp_server_registers_viewer_tools_without_bridge(monkeypatch):
    proxy = _FakeProxy()
    calls = []

    def fake_create_remote_proxy(**kwargs):
        calls.append(kwargs)
        return mcp_server.RemoteProxy(proxy=proxy, tool_factory=_FakeToolFactory())

    monkeypatch.setattr(mcp_server, "_create_remote_proxy", fake_create_remote_proxy)
    monkeypatch.setattr(mcp_server, "_resolve_api_key", lambda: "api-key")
    monkeypatch.setattr(mcp_server, "_resolve_requested_bridge", lambda explicit: None)

    mcp_server.run_mcp_server()

    assert calls[0]["api_key"] == "api-key"
    assert proxy.tools == [
        "detect_python_environment",
        "validate_simulation",
        "rotate_viewer",
        "capture",
        "show_structures",
    ]
    assert proxy.ran is True


def test_run_mcp_server_rejects_invalid_explicit_viewer_bridge(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "_create_remote_proxy",
        lambda **_kwargs: pytest.fail("proxy should not be created with invalid bridge input"),
    )
    monkeypatch.setattr(mcp_server, "_resolve_api_key", lambda: "api-key")

    with pytest.raises(RuntimeError, match="Invalid --viewer-bridge value"):
        mcp_server.run_mcp_server(viewer_bridge="cursor://Flexcompute.tidy3d/bridge")


def test_run_mcp_server_rejects_invalid_bridge_env_var(monkeypatch):
    monkeypatch.setenv("TIDY3D_VIEWER_BRIDGE_URL", "https://example.com:5123")
    monkeypatch.setattr(
        mcp_server,
        "_create_remote_proxy",
        lambda **_kwargs: pytest.fail("proxy should not be created with invalid bridge env"),
    )
    monkeypatch.setattr(mcp_server, "_resolve_api_key", lambda: "api-key")

    with pytest.raises(RuntimeError, match="Invalid TIDY3D_VIEWER_BRIDGE_URL value"):
        mcp_server.run_mcp_server()


def test_run_mcp_server_uses_namespaced_remote_url_env(monkeypatch):
    proxy = _FakeProxy()
    calls = []

    def fake_create_remote_proxy(**kwargs):
        calls.append(kwargs)
        return mcp_server.RemoteProxy(proxy=proxy, tool_factory=_FakeToolFactory())

    monkeypatch.setenv("REMOTE_MCP_URL", "https://ignored.example/")
    monkeypatch.setenv("TIDY3D_MCP_REMOTE_URL", "https://remote.example/")
    monkeypatch.setattr(mcp_server, "_create_remote_proxy", fake_create_remote_proxy)
    monkeypatch.setattr(mcp_server, "_resolve_api_key", lambda: "api-key")
    monkeypatch.setattr(mcp_server, "_resolve_requested_bridge", lambda explicit: None)

    mcp_server.run_mcp_server()

    assert calls[0]["mcp_url"] == "https://remote.example/"


def test_resolve_api_key_uses_tidy3d_configuration(monkeypatch):
    monkeypatch.setattr(mcp_server, "configured_api_key", lambda: "configured-api-key")

    assert mcp_server._resolve_api_key() == "configured-api-key"


def test_resolve_api_key_reports_tidy3d_configuration_steps(monkeypatch):
    monkeypatch.setattr(mcp_server, "configured_api_key", lambda: None)

    with pytest.raises(RuntimeError, match="Set SIMCLOUD_APIKEY or run `tidy3d configure`"):
        mcp_server._resolve_api_key()


def test_module_entrypoint_forwards_cli_options(monkeypatch):
    calls = []

    def fake_run_mcp_server(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(mcp_server, "run_mcp_server", fake_run_mcp_server)

    mcp_server.main(["--viewer-bridge", ":5123"])

    assert calls == [{"viewer_bridge": ":5123"}]


class _FakeProxy:
    def __init__(self):
        self.tools = []
        self.ran = False

    def add_tool(self, tool):
        self.tools.append(tool)

    def run(self, *, show_banner):
        assert show_banner is False
        self.ran = True


class _FakeToolFactory:
    def from_function(self, function):
        return function.__name__

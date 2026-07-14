"""Tests for bundled Tidy3D MCP server helpers."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from tidy3d.web.mcp import _dispatcher, python_env, screenshots, viewer
from tidy3d.web.mcp import server as mcp_server

_FASTMCP_SERVER_SMOKE_HELPERS = """
import contextlib
import threading
import time


def server_bound_port(server):
    for uvicorn_server in getattr(server, "servers", []) or []:
        for sock in getattr(uvicorn_server, "sockets", []) or []:
            port = sock.getsockname()[1]
            if isinstance(port, int):
                return port
    return None


@contextlib.contextmanager
def run_streamable_http_server(upstream, *, path="/mcp", received_headers=None):
    import uvicorn

    upstream_app = upstream.http_app(transport="streamable-http", path=path)
    app = upstream_app
    if received_headers is not None:

        async def capture_headers(scope, receive, send):
            if scope["type"] == "http":
                received_headers.append(
                    {
                        key.decode("latin-1"): value.decode("latin-1")
                        for key, value in scope["headers"]
                    }
                )
            await upstream_app(scope, receive, send)

        app = capture_headers

    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=0,
            lifespan="on",
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    port = None
    deadline = time.monotonic() + 10
    while thread.is_alive() and time.monotonic() < deadline:
        if server.started:
            port = server_bound_port(server)
            if port is not None:
                break
        time.sleep(0.05)
    if port is None:
        raise RuntimeError("FastMCP streamable HTTP test server did not start")

    try:
        if path == "/":
            yield f"http://127.0.0.1:{port}/"
        else:
            yield f"http://127.0.0.1:{port}{path}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        if thread.is_alive():
            raise RuntimeError("FastMCP streamable HTTP test server did not stop")
"""


def _run_python_smoke(script: str) -> None:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{Path.cwd()}{os.pathsep}{pythonpath}" if pythonpath else str(Path.cwd())
    subprocess_script = f"{_FASTMCP_SERVER_SMOKE_HELPERS}\n{textwrap.dedent(script)}"
    result = subprocess.run(
        [sys.executable, "-c", subprocess_script],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, (
        f"MCP smoke subprocess failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


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


def test_viewer_file_uri_uses_windows_path_conversion(monkeypatch):
    converted_paths = []

    def fake_windows_url2pathname(url_path):
        converted_paths.append(url_path)
        return f"converted:{url_path}"

    monkeypatch.setattr(viewer.nturl2path, "url2pathname", fake_windows_url2pathname)

    assert (
        viewer._local_file_path_text("file:///C:/Users/Ada/simulation.py", os_name="nt")
        == "converted:/C:/Users/Ada/simulation.py"
    )
    assert (
        viewer._local_file_path_text("file://localhost/C:/Users/Ada/simulation.py", os_name="nt")
        == "converted:/C:/Users/Ada/simulation.py"
    )
    assert (
        viewer._local_file_path_text("file://server/share/simulation.py", os_name="nt")
        == "converted://server/share/simulation.py"
    )
    assert converted_paths == [
        "/C:/Users/Ada/simulation.py",
        "/C:/Users/Ada/simulation.py",
        "//server/share/simulation.py",
    ]


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


def test_run_mcp_server_smoke_uses_real_fastmcp_proxy():
    _run_python_smoke("""
        import asyncio
        import os

        from fastmcp import Client
        from fastmcp.server.providers.proxy import FastMCPProxy, ProxyProvider
        from tidy3d.web.mcp import _dispatcher, python_env
        from tidy3d.web.mcp import server as mcp_server

        run_calls = []

        async def no_remote_tool(_self, _name, version=None):
            return None

        async def no_remote_tools(_self):
            return []

        def fake_run(self, *, show_banner):
            async def client_discovery_and_call():
                async with Client(self) as client:
                    tools = await client.list_tools()
                    result = await client.call_tool("detect_python_environment", {"resource": None})
                return tools, result

            tools, result = asyncio.run(client_discovery_and_call())
            run_calls.append((self, show_banner, tools, result))

        ProxyProvider.get_tool = no_remote_tool
        ProxyProvider.list_tools = no_remote_tools
        FastMCPProxy.run = fake_run
        _dispatcher._BRIDGE_URL = None
        mcp_server.configured_api_key = lambda: "api-key"
        python_env.detect_python_environment_payload = lambda resource=None: {
            "pythonExec": "/env/bin/python",
            "envManager": "venv",
            "projectManager": "uv",
            "detectionSource": "workspace",
        }
        os.environ.pop(mcp_server.REMOTE_MCP_URL_ENV, None)
        os.environ["TIDY3D_MCP_USER_AGENT"] = "tidy3d-test"

        mcp_server.run_mcp_server(viewer_bridge=":5123")

        assert len(run_calls) == 1
        proxy, show_banner, tools, result = run_calls[0]
        assert isinstance(proxy, FastMCPProxy)
        assert show_banner is False
        assert {tool.name for tool in tools} == {
            "detect_python_environment",
            "validate_simulation",
            "rotate_viewer",
            "capture",
            "show_structures",
        }
        assert result.structured_content == {
            "pythonExec": "/env/bin/python",
            "envManager": "venv",
            "projectManager": "uv",
            "detectionSource": "workspace",
        }
        assert result.content[0].text == (
            "interpreter: /env/bin/python; env manager: venv; project manager: uv "
            "(source=workspace)"
        )
    """)


def test_create_remote_proxy_smoke_invokes_proxied_fastmcp_tool():
    _run_python_smoke("""
        import asyncio

        from fastmcp import Client, FastMCP
        from tidy3d.web.mcp import server as mcp_server

        upstream = FastMCP("Upstream")

        @upstream.tool
        async def upstream_status() -> dict[str, str]:
            return {"status": "ok"}

        received_headers = []
        with run_streamable_http_server(upstream, received_headers=received_headers) as mcp_url:
            remote = mcp_server._create_remote_proxy(
                api_key="api-key",
                mcp_url=mcp_url,
                user_agent="tidy3d-test",
            )

            async def client_discovery_and_call():
                async with Client(remote.proxy) as client:
                    tools = await client.list_tools()
                    result = await client.call_tool("upstream_status", {})
                return tools, result

            tools, result = asyncio.run(client_discovery_and_call())

        assert {tool.name for tool in tools} == {"upstream_status"}
        assert result.structured_content == {"status": "ok"}
        assert any(headers.get("user-agent") == "tidy3d-test" for headers in received_headers)
        assert any(headers.get("authorization") == "Bearer api-key" for headers in received_headers)
    """)


@pytest.mark.parametrize("url_has_trailing_slash", [False, True])
def test_create_remote_proxy_smoke_invokes_root_mounted_fastmcp_tool(url_has_trailing_slash):
    _run_python_smoke(f"""
        import asyncio

        from fastmcp import Client, FastMCP
        from tidy3d.web.mcp import server as mcp_server

        url_has_trailing_slash = {url_has_trailing_slash!r}
        upstream = FastMCP("RootMountedUpstream")

        @upstream.tool
        async def root_status() -> dict[str, str]:
            return {{"status": "root-ok"}}

        with run_streamable_http_server(upstream, path="/") as root_url:
            mcp_url = root_url if url_has_trailing_slash else root_url.rstrip("/")
            remote = mcp_server._create_remote_proxy(
                api_key="api-key",
                mcp_url=mcp_url,
                user_agent="tidy3d-test",
            )

            async def client_discovery_and_call():
                async with Client(remote.proxy) as client:
                    tools = await client.list_tools()
                    result = await client.call_tool("root_status", {{}})
                return tools, result

            tools, result = asyncio.run(client_discovery_and_call())

        assert {{tool.name for tool in tools}} == {{"root_status"}}
        assert result.structured_content == {{"status": "root-ok"}}
    """)


def test_mcp_stdio_server_smoke_uses_real_startup_path():
    _run_python_smoke("""
        import asyncio
        import os
        import sys
        import tempfile
        from pathlib import Path

        from fastmcp import Client, FastMCP
        from fastmcp.client.transports import StdioTransport
        from tidy3d.web.mcp import server as mcp_server

        upstream = FastMCP("Upstream")

        @upstream.tool
        async def upstream_status() -> dict[str, str]:
            return {"status": "ok"}

        with run_streamable_http_server(upstream) as mcp_url:
            env = os.environ.copy()
            env.update(
                {
                    "SIMCLOUD_APIKEY": "api-key",
                    mcp_server.REMOTE_MCP_URL_ENV: mcp_url,
                    "TIDY3D_MCP_USER_AGENT": "tidy3d-test",
                }
            )
            env.pop("TIDY3D_VIEWER_BRIDGE_URL", None)

            with tempfile.TemporaryDirectory() as tmp_dir:
                transport = StdioTransport(
                    command=sys.executable,
                    args=["-m", "tidy3d.web.mcp.server"],
                    env=env,
                    cwd=os.getcwd(),
                    keep_alive=False,
                    log_file=Path(tmp_dir) / "mcp-server.log",
                )

                async def client_discovery_and_call():
                    async with Client(transport) as client:
                        tools = await client.list_tools()
                        result = await client.call_tool("upstream_status", {})
                    return tools, result

                tools, result = asyncio.run(client_discovery_and_call())

        assert {tool.name for tool in tools} == {
            "capture",
            "detect_python_environment",
            "rotate_viewer",
            "show_structures",
            "upstream_status",
            "validate_simulation",
        }
        assert result.structured_content == {"status": "ok"}
    """)


def test_default_remote_mcp_url_matches_hosted_flexagent_root():
    from urllib.parse import urlsplit

    parsed = urlsplit(mcp_server.DEFAULT_REMOTE_MCP_URL)

    assert parsed.scheme == "https"
    assert parsed.netloc == "flexagent.simulation.cloud"
    assert parsed.path == "/"
    assert parsed.query == ""


def test_run_mcp_server_uses_checked_in_default_remote_url(monkeypatch):
    proxy = _FakeProxy()
    calls = []

    def fake_create_remote_proxy(**kwargs):
        calls.append(kwargs)
        return mcp_server.RemoteProxy(proxy=proxy, tool_factory=_FakeToolFactory())

    monkeypatch.delenv(mcp_server.REMOTE_MCP_URL_ENV, raising=False)
    monkeypatch.setattr(mcp_server, "_create_remote_proxy", fake_create_remote_proxy)
    monkeypatch.setattr(mcp_server, "_resolve_api_key", lambda: "api-key")
    monkeypatch.setattr(mcp_server, "_resolve_requested_bridge", lambda explicit: None)

    mcp_server.run_mcp_server()

    assert calls[0]["mcp_url"] == mcp_server.DEFAULT_REMOTE_MCP_URL


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
    monkeypatch.setenv("TIDY3D_MCP_REMOTE_URL", "https://remote.example/mcp")
    monkeypatch.setattr(mcp_server, "_create_remote_proxy", fake_create_remote_proxy)
    monkeypatch.setattr(mcp_server, "_resolve_api_key", lambda: "api-key")
    monkeypatch.setattr(mcp_server, "_resolve_requested_bridge", lambda explicit: None)

    mcp_server.run_mcp_server()

    assert calls[0]["mcp_url"] == "https://remote.example/mcp"


@pytest.mark.parametrize(
    "hosted_url",
    [
        "https://flexagent.simulation.cloud",
        "https://flexagent.simulation.cloud/",
    ],
)
def test_run_mcp_server_preserves_hosted_remote_url_env(monkeypatch, hosted_url):
    proxy = _FakeProxy()
    calls = []

    def fake_create_remote_proxy(**kwargs):
        calls.append(kwargs)
        return mcp_server.RemoteProxy(proxy=proxy, tool_factory=_FakeToolFactory())

    monkeypatch.setenv(mcp_server.REMOTE_MCP_URL_ENV, hosted_url)
    monkeypatch.setattr(mcp_server, "_create_remote_proxy", fake_create_remote_proxy)
    monkeypatch.setattr(mcp_server, "_resolve_api_key", lambda: "api-key")
    monkeypatch.setattr(mcp_server, "_resolve_requested_bridge", lambda explicit: None)

    mcp_server.run_mcp_server()

    assert calls[0]["mcp_url"] == hosted_url


@pytest.mark.parametrize(
    "remote_url",
    [
        "https://internal.example",
        "https://internal.example/",
        "https://internal.example/mcp",
        "https://internal.example/mcp/",
    ],
)
def test_run_mcp_server_preserves_custom_remote_url_env_exactly(monkeypatch, remote_url):
    proxy = _FakeProxy()
    calls = []

    def fake_create_remote_proxy(**kwargs):
        calls.append(kwargs)
        return mcp_server.RemoteProxy(proxy=proxy, tool_factory=_FakeToolFactory())

    monkeypatch.setenv(mcp_server.REMOTE_MCP_URL_ENV, remote_url)
    monkeypatch.setattr(mcp_server, "_create_remote_proxy", fake_create_remote_proxy)
    monkeypatch.setattr(mcp_server, "_resolve_api_key", lambda: "api-key")
    monkeypatch.setattr(mcp_server, "_resolve_requested_bridge", lambda explicit: None)

    mcp_server.run_mcp_server()

    assert calls[0]["mcp_url"] == remote_url


@pytest.mark.parametrize(
    "remote_url",
    [
        "/mcp",
        "internal.example/mcp",
        "ftp://host.example/mcp",
    ],
)
def test_run_mcp_server_rejects_non_http_or_relative_remote_url_env(monkeypatch, remote_url):
    monkeypatch.setenv(mcp_server.REMOTE_MCP_URL_ENV, remote_url)
    monkeypatch.setattr(
        mcp_server,
        "_create_remote_proxy",
        lambda **_kwargs: pytest.fail("proxy should not be created with invalid remote URL"),
    )
    monkeypatch.setattr(mcp_server, "_resolve_api_key", lambda: "api-key")
    monkeypatch.setattr(mcp_server, "_resolve_requested_bridge", lambda explicit: None)

    with pytest.raises(RuntimeError, match="root-mounted endpoint") as exc_info:
        mcp_server.run_mcp_server()

    message = str(exc_info.value)
    assert mcp_server.REMOTE_MCP_URL_SETTING in message
    assert mcp_server.REMOTE_MCP_URL_ENV in message


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

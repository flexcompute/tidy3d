from __future__ import annotations

import ssl

import pytest

from tidy3d.web import diagnostics
from tidy3d.web.core import http_util


class _FakeResponse:
    def __init__(
        self,
        chunks: tuple[bytes, ...] = (),
        json_payload=None,
        json_exception: Exception | None = None,
        headers: dict[str, str] | None = None,
        url: str | None = None,
    ) -> None:
        self.ok = True
        self.chunks = chunks
        self.closed = False
        self.json_payload = json_payload
        self.json_exception = json_exception
        self.headers = headers or {}
        self.url = url

    def raise_for_status(self) -> None:
        raise AssertionError("raise_for_status should not be called for successful responses")

    def json(self):
        if self.json_exception is not None:
            raise self.json_exception
        return self.json_payload

    def iter_content(self, chunk_size: int):
        yield from self.chunks

    def close(self) -> None:
        self.closed = True


class _FakeSslContext:
    def __init__(self) -> None:
        self.load_verify_locations_calls = []

    def load_verify_locations(self, *, cafile=None, capath=None) -> None:
        self.load_verify_locations_calls.append((cafile, capath))


class _FakeSession:
    download_url = "https://storage.example.com/diagnostic.bin?X-Amz-Signature=secret"
    download_chunk = b"x" * diagnostics.DOWNLOAD_CHUNK_SIZE

    def __init__(self, download_chunks: tuple[bytes, ...] | None = None) -> None:
        self.calls = []
        self.responses = []
        self.download_chunks = download_chunks or (self.download_chunk,) * (
            diagnostics.DEFAULT_DOWNLOAD_BYTES // diagnostics.DOWNLOAD_CHUNK_SIZE
        )

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith(diagnostics.DEFAULT_DOWNLOAD_URL_ENDPOINT):
            response = _FakeResponse(json_payload={"downloadUrl": self.download_url}, url=url)
        elif url.endswith("/health"):
            response = _FakeResponse(json_payload={"health": "OK"}, url=url)
        elif url.endswith("tidy3d/projects"):
            response = _FakeResponse(json_payload={"data": []}, url=url)
        else:
            response = _FakeResponse(
                chunks=self.download_chunks if kwargs.get("stream") else (),
                url=url,
            )
        self.responses.append(response)
        return response


def _patch_network_path(monkeypatch):
    monkeypatch.setattr(
        diagnostics,
        "_check_network_path",
        lambda *,
        api_url,
        timeout,
        include_private_network_details=False: diagnostics.ConnectionDiagnosticResult(
            name="network_path",
            status="pass",
            target_host="api.example.com",
            detail="direct dns 1.0 ms, tcp 2.0 ms, tls TLSv1.3",
            metadata={"mode": "direct_socket", "resolved_ip_count": 1, "tls_version": "TLSv1.3"},
        ),
    )


class _FailingDownloadSession(_FakeSession):
    def get(self, url, **kwargs):
        if kwargs.get("stream"):
            self.calls.append((url, kwargs))
            raise RuntimeError(f"failed to download {url}")
        return super().get(url, **kwargs)


class _RelativeUrlFailingDownloadSession(_FakeSession):
    def get(self, url, **kwargs):
        if kwargs.get("stream"):
            self.calls.append((url, kwargs))
            raise RuntimeError(
                "Max retries exceeded with url: "
                "/diagnostic.bin?X-Amz-Signature=secret&X-Amz-Credential=credential"
            )
        return super().get(url, **kwargs)


class _BlockedHealthSession(_FakeSession):
    def get(self, url, **kwargs):
        if url.endswith("/health"):
            self.calls.append((url, kwargs))
            response = _FakeResponse(
                json_exception=ValueError("Expecting value"),
                headers={"content-type": "text/html"},
                url=url,
            )
            self.responses.append(response)
            return response
        return super().get(url, **kwargs)


class _BlockedAuthenticationSession(_FakeSession):
    def get(self, url, **kwargs):
        if url.endswith("tidy3d/projects"):
            self.calls.append((url, kwargs))
            response = _FakeResponse(
                json_exception=ValueError("Expecting value"),
                headers={"content-type": "text/html"},
                url=url,
            )
            self.responses.append(response)
            return response
        return super().get(url, **kwargs)


class _UnexpectedAuthenticationJsonSession(_FakeSession):
    def get(self, url, **kwargs):
        if url.endswith("tidy3d/projects"):
            self.calls.append((url, kwargs))
            response = _FakeResponse(
                json_payload={"error": "blocked by proxy"},
                headers={"content-type": "application/json"},
                url=url,
            )
            self.responses.append(response)
            return response
        return super().get(url, **kwargs)


class _CertificateFailureSession(_FakeSession):
    def get(self, url, **kwargs):
        if url.endswith("/health"):
            self.calls.append((url, kwargs))
            raise diagnostics.requests.exceptions.SSLError(
                ssl.SSLCertVerificationError(
                    1,
                    "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
                    "unable to get local issuer certificate",
                )
            )
        return super().get(url, **kwargs)


def test_diagnose_connection_uses_default_storage_download_url(monkeypatch):
    """Diagnostic report should always resolve and use the default storage test URL."""
    session = _FakeSession()
    monkeypatch.setattr(diagnostics.requests, "Session", lambda: session)
    monkeypatch.setattr(diagnostics, "api_key", lambda: "configured")
    _patch_network_path(monkeypatch)

    report = diagnostics.diagnose_connection(
        api_samples=2,
        verbose=False,
    )

    assert report.api_key_configured is True
    assert report.privacy_mode == "shareable"
    assert report.private_details_warning is None
    assert [result.name for result in report.results] == [
        "network_path",
        "api_latency",
        "authentication",
        "storage_download",
    ]
    assert report.results[0].status == "pass"
    assert report.results[0].metadata["tls_version"] == "TLSv1.3"
    assert report.results[1].status == "pass"
    assert len(report.results[1].samples) == 2
    assert report.results[2].status == "pass"
    assert report.results[3].status == "pass"
    assert report.results[3].target_host == "storage.example.com"
    assert session.calls[-2][0].endswith(diagnostics.DEFAULT_DOWNLOAD_URL_ENDPOINT)
    assert session.calls[-1][0] == _FakeSession.download_url
    assert len(session.calls) == 5


@pytest.mark.parametrize(
    ("session_type", "result_index", "expected_error"),
    (
        (
            _BlockedHealthSession,
            1,
            "Health endpoint returned a non-JSON response "
            "(content-type: text_html; parse error: Expecting value).",
        ),
        (
            _BlockedAuthenticationSession,
            2,
            "Authentication endpoint returned a non-JSON response "
            "(content-type: text_html; parse error: Expecting value).",
        ),
        (
            _UnexpectedAuthenticationJsonSession,
            2,
            "Authentication endpoint returned unexpected JSON payload.",
        ),
    ),
)
def test_diagnose_connection_rejects_false_positive_200s(
    monkeypatch, session_type, result_index, expected_error
):
    """HTTP 200 should not pass unless the API response shape is recognizable."""

    session = session_type()
    monkeypatch.setattr(diagnostics.requests, "Session", lambda: session)
    monkeypatch.setattr(diagnostics, "api_key", lambda: "configured")
    _patch_network_path(monkeypatch)

    report = diagnostics.diagnose_connection(
        api_samples=1,
        verbose=False,
    )

    result = report.results[result_index]
    assert result.status == "fail"
    assert result.error_type == "ValueError"
    assert result.error == expected_error


def test_diagnose_connection_recommends_managed_network_ca_fix(monkeypatch):
    """Certificate verification failures should point to network CA remedies."""

    session = _CertificateFailureSession()
    monkeypatch.setattr(diagnostics.requests, "Session", lambda: session)
    monkeypatch.setattr(diagnostics, "api_key", lambda: "configured")
    _patch_network_path(monkeypatch)

    report = diagnostics.diagnose_connection(
        api_samples=1,
        verbose=False,
    )

    api_result = report.results[1]
    assert api_result.status == "fail"
    assert api_result.error_type == "SSLError"
    assert "certificate verify failed" in api_result.error
    assert (
        "python -m pip install --trusted-host pypi.org --trusted-host "
        "files.pythonhosted.org pip-system-certs"
    ) in api_result.recommendation
    assert "managed networks" in api_result.recommendation
    assert "REQUESTS_CA_BUNDLE or CURL_CA_BUNDLE" in api_result.recommendation
    assert "allow access to tidy3d-api.simulation.cloud" in api_result.recommendation
    assert "recommendation: Python certificate verification failed" in report.support_text()


def test_diagnose_connection_measures_storage_download_and_redacts_url(monkeypatch):
    """Storage throughput should read bytes while keeping signed URL secrets out of reports."""
    session = _FakeSession()
    monkeypatch.setattr(diagnostics.requests, "Session", lambda: session)
    monkeypatch.setattr(diagnostics, "api_key", lambda: "configured")
    _patch_network_path(monkeypatch)

    report = diagnostics.diagnose_connection(
        api_samples=1,
        verbose=False,
    )

    storage_result = report.results[3]
    assert storage_result.status == "pass"
    assert storage_result.target_host == "storage.example.com"
    assert storage_result.samples[0].bytes_transferred == diagnostics.DEFAULT_DOWNLOAD_BYTES
    assert storage_result.samples[0].throughput_mib_s > 0
    assert session.calls[-1][1]["headers"] == {"Range": "bytes=0-104857599"}
    assert session.responses[-1].closed is True
    assert "secret" not in report.model_dump_json()
    assert "X-Amz-Signature" not in report.support_text()


def test_diagnose_connection_fails_on_short_storage_download(monkeypatch):
    """Storage throughput should not pass when fewer bytes arrive than requested."""
    session = _FakeSession(download_chunks=(_FakeSession.download_chunk,) * 3)
    monkeypatch.setattr(diagnostics.requests, "Session", lambda: session)
    monkeypatch.setattr(diagnostics, "api_key", lambda: "configured")
    _patch_network_path(monkeypatch)

    report = diagnostics.diagnose_connection(
        api_samples=1,
        verbose=False,
    )

    storage_result = report.results[3]
    assert storage_result.status == "fail"
    assert storage_result.error_type == "ValueError"
    assert storage_result.error == (
        "Diagnostic storage download returned 3145728 bytes, expected 104857600 bytes."
    )
    assert session.responses[-1].closed is True


def test_diagnose_connection_redacts_signed_url_from_errors(monkeypatch):
    """Signed URL query strings should not leak through exception messages."""
    session = _FailingDownloadSession()
    monkeypatch.setattr(diagnostics.requests, "Session", lambda: session)
    monkeypatch.setattr(diagnostics, "api_key", lambda: "configured")
    _patch_network_path(monkeypatch)

    report = diagnostics.diagnose_connection(
        api_samples=1,
        verbose=False,
    )

    storage_result = report.results[3]
    assert storage_result.status == "fail"
    assert storage_result.error == "failed to download https://storage.example.com/diagnostic.bin"
    assert "secret" not in report.model_dump_json()


def test_diagnose_connection_redacts_relative_signed_url_from_errors(monkeypatch):
    """urllib3-style relative URL errors should not expose signed URL query strings."""
    session = _RelativeUrlFailingDownloadSession()
    monkeypatch.setattr(diagnostics.requests, "Session", lambda: session)
    monkeypatch.setattr(diagnostics, "api_key", lambda: "configured")
    _patch_network_path(monkeypatch)

    report = diagnostics.diagnose_connection(
        api_samples=1,
        verbose=False,
    )

    storage_result = report.results[3]
    assert storage_result.status == "fail"
    assert storage_result.error == "Max retries exceeded with url: /diagnostic.bin"
    assert "secret" not in report.model_dump_json()
    assert "X-Amz-Signature" not in report.support_text()


def test_configured_session_applies_client_ssl_settings(monkeypatch):
    """Diagnostics should use the same SSL verification and TLS adapter knobs as the client."""

    class MountableSession(_FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.mounted = []

        def mount(self, prefix, adapter) -> None:
            self.mounted.append((prefix, adapter))

    session = MountableSession()
    monkeypatch.setattr(diagnostics.requests, "Session", lambda: session)
    monkeypatch.setattr(diagnostics.config.web, "ssl_verify", False)
    monkeypatch.setattr(diagnostics.config.web, "ssl_version", "TLSv1_2")

    configured = diagnostics._configured_session()

    assert configured is session
    assert session.verify is False
    assert session.mounted[0][0] == "https://"
    assert isinstance(session.mounted[0][1], diagnostics.TLSAdapter)


def test_shared_ssl_context_uses_configured_tls_version(monkeypatch):
    """HTTP and diagnostics TLS probes should share TLS-version interpretation."""

    calls = []
    context = _FakeSslContext()

    def fake_create_urllib3_context(*, ssl_version=None, cert_reqs=None):
        calls.append((ssl_version, cert_reqs))
        return context

    monkeypatch.setattr(http_util, "create_urllib3_context", fake_create_urllib3_context)
    monkeypatch.setattr(http_util.config.web, "ssl_version", "TLSv1_3")

    result = http_util.ssl_context_for_config(cert_reqs=ssl.CERT_NONE)

    assert result is context
    assert calls == [(ssl.TLSVersion.TLSv1_3, ssl.CERT_NONE)]


@pytest.mark.parametrize(
    (
        "ssl_verify",
        "environment",
        "requests_default",
        "is_directory",
        "expected_cert_reqs",
        "expected_load_verify_locations",
    ),
    (
        (
            True,
            {"REQUESTS_CA_BUNDLE": "/corp/request-ca.pem", "CURL_CA_BUNDLE": "/corp/curl-ca.pem"},
            "/requests/cacert.pem",
            False,
            ssl.CERT_REQUIRED,
            [("/corp/request-ca.pem", None)],
        ),
        (
            True,
            {},
            "/requests/cacert.pem",
            False,
            ssl.CERT_REQUIRED,
            [("/requests/cacert.pem", None)],
        ),
        (
            True,
            {"CURL_CA_BUNDLE": "/corp/certs"},
            "/requests/cacert.pem",
            True,
            ssl.CERT_REQUIRED,
            [(None, "/corp/certs")],
        ),
        (
            False,
            {"REQUESTS_CA_BUNDLE": "/corp/request-ca.pem"},
            "/requests/cacert.pem",
            False,
            ssl.CERT_NONE,
            [],
        ),
    ),
)
def test_diagnostic_ssl_context_applies_verification_and_ca_source(
    monkeypatch,
    ssl_verify,
    environment,
    requests_default,
    is_directory,
    expected_cert_reqs,
    expected_load_verify_locations,
):
    """Direct TLS probes add Requests CA behavior around the shared SSL context."""

    calls = []
    context = _FakeSslContext()

    def fake_ssl_context_for_config(*, cert_reqs=None):
        calls.append(cert_reqs)
        return context

    monkeypatch.setattr(diagnostics, "ssl_context_for_config", fake_ssl_context_for_config)
    monkeypatch.setattr(diagnostics.config.web, "ssl_verify", ssl_verify)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    monkeypatch.delenv("CURL_CA_BUNDLE", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(diagnostics.requests.certs, "where", lambda: requests_default)
    monkeypatch.setattr(diagnostics.os.path, "isdir", lambda path: is_directory)

    result = diagnostics._ssl_context_for_config()

    assert result is context
    assert calls == [expected_cert_reqs]
    assert context.load_verify_locations_calls == expected_load_verify_locations


def test_network_path_reports_dns_tcp_tls_metadata(monkeypatch):
    """Network path check should split DNS, TCP, and TLS details for support."""

    class FakeSocket:
        instances = []

        def __init__(self, family, socktype, proto) -> None:
            self.family = family
            self.socktype = socktype
            self.proto = proto
            self.timeout = None
            self.connected_to = None
            self.closed = False
            self.instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def settimeout(self, timeout) -> None:
            self.timeout = timeout

        def connect(self, sockaddr) -> None:
            self.connected_to = sockaddr

        def close(self) -> None:
            self.closed = True

        def getpeername(self):
            return ("203.0.113.10", 443)

    class FakeTlsSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            return None

        def version(self):
            return "TLSv1.3"

        def cipher(self):
            return ("TLS_AES_128_GCM_SHA256", "TLSv1.3", 128)

        def getpeercert(self, binary_form=False):
            if binary_form:
                return b"certificate"
            return {
                "subject": ((("commonName", "*.simulation.cloud"),),),
                "issuer": ((("organizationName", "Amazon"),),),
                "notAfter": "Sep 14 23:59:59 2026 GMT",
            }

    class FakeContext:
        def wrap_socket(self, sock, server_hostname):
            assert server_hostname == "api.example.com"
            return FakeTlsSocket()

    monkeypatch.setattr(
        diagnostics.socket,
        "getaddrinfo",
        lambda host, port, type: [
            (
                diagnostics.socket.AF_INET,
                diagnostics.socket.SOCK_STREAM,
                6,
                "",
                ("203.0.113.10", port),
            ),
            (
                diagnostics.socket.AF_INET,
                diagnostics.socket.SOCK_STREAM,
                6,
                "",
                ("203.0.113.10", port),
            ),
            (
                diagnostics.socket.AF_INET,
                diagnostics.socket.SOCK_STREAM,
                6,
                "",
                ("203.0.113.11", port),
            ),
        ],
    )
    monkeypatch.setattr(diagnostics.socket, "socket", FakeSocket)
    monkeypatch.setattr(
        diagnostics.socket,
        "create_connection",
        lambda target, timeout: (_ for _ in ()).throw(
            AssertionError("network_path should use resolved addr_info")
        ),
    )
    monkeypatch.setattr(diagnostics, "_ssl_context_for_config", lambda: FakeContext())

    result = diagnostics._check_network_path(
        api_url="https://api.example.com:8443",
        timeout=3.0,
    )

    assert result.name == "network_path"
    assert result.status == "pass"
    assert result.target_host == "api.example.com:8443"
    assert result.metadata["mode"] == "direct_socket"
    assert result.metadata["port"] == 8443
    assert result.metadata["resolved_ip_count"] == 2
    assert result.metadata["tls_version"] == "TLSv1.3"
    assert result.metadata["certificate_present"] is True
    assert FakeSocket.instances[0].family == diagnostics.socket.AF_INET
    assert FakeSocket.instances[0].socktype == diagnostics.socket.SOCK_STREAM
    assert FakeSocket.instances[0].proto == 6
    assert FakeSocket.instances[0].timeout == 3.0
    assert FakeSocket.instances[0].connected_to == ("203.0.113.10", 8443)
    assert "resolved_ips" not in result.metadata
    assert "remote_ip" not in result.metadata
    assert "certificate_subject" not in result.metadata
    assert "certificate_issuer" not in result.metadata
    assert "direct dns" in result.detail
    assert "tls TLSv1.3" in result.detail

    private_result = diagnostics._check_network_path(
        api_url="https://api.example.com:8443",
        timeout=3.0,
        include_private_network_details=True,
    )

    assert private_result.metadata["resolved_ips"] == ("203.0.113.10", "203.0.113.11")
    assert private_result.metadata["remote_ip"] == "203.0.113.10"
    assert private_result.metadata["certificate_subject"] == "commonName=*.simulation.cloud"
    assert private_result.metadata["certificate_issuer"] == "organizationName=Amazon"
    assert private_result.metadata["certificate_not_after"] == "Sep 14 23:59:59 2026 GMT"


def test_host_from_url_strips_userinfo() -> None:
    """Host extraction should not expose credentials embedded in custom URLs."""

    assert (
        diagnostics._host_from_url("https://user:secret@api.example.com:8443/path?token=x")
        == "api.example.com:8443"
    )


def test_diagnostic_configuration_redacts_env_and_warns_on_legacy_ssl(monkeypatch):
    """Support configuration should include useful state without private network values."""

    monkeypatch.setattr(
        diagnostics.config.web,
        "api_endpoint",
        "https://user:secret@[2001:db8::1]:8443/api?token=x",
    )
    monkeypatch.setenv("HTTPS_PROXY", "http://user:secret@[2001:db8::2]:8080")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/tmp/cert.pem")
    monkeypatch.setenv(
        "TIDY3D_WEB__API_ENDPOINT", "https://user:secret@[2001:db8::3]:9443/api?token=x"
    )
    monkeypatch.setenv("TIDY3D_WEB__SSL_VERIFY", "false")
    monkeypatch.setenv("TIDY3D_SSL_VERIFY", "false")

    configuration = diagnostics._diagnostic_configuration()

    assert configuration.api_endpoint == "https://[2001:db8::1]:8443/api"
    assert configuration.proxy_environment["HTTPS_PROXY"] == "set"
    assert configuration.certificate_environment["REQUESTS_CA_BUNDLE"] == "set"
    assert configuration.tidy3d_environment["TIDY3D_WEB__API_ENDPOINT"] == (
        "https://[2001:db8::3]:9443/api"
    )
    assert configuration.tidy3d_environment["TIDY3D_WEB__SSL_VERIFY"] == "false"
    assert configuration.tidy3d_environment["TIDY3D_SSL_VERIFY"] == "false"
    assert configuration.warnings == (
        "TIDY3D_SSL_VERIFY is set but ignored; use TIDY3D_WEB__SSL_VERIFY to configure "
        "SSL verification.",
    )
    assert "http://[2001:db8::2]:8080" not in configuration.model_dump_json()
    assert "/tmp/cert.pem" not in configuration.model_dump_json()
    assert "secret" not in configuration.model_dump_json()
    assert "token=x" not in configuration.model_dump_json()

    private_configuration = diagnostics._diagnostic_configuration(
        include_private_network_details=True
    )

    assert private_configuration.api_endpoint == "https://[2001:db8::1]:8443/api"
    assert private_configuration.tidy3d_environment["TIDY3D_WEB__API_ENDPOINT"] == (
        "https://[2001:db8::3]:9443/api"
    )
    assert private_configuration.proxy_environment["HTTPS_PROXY"] == ("http://[2001:db8::2]:8080")
    assert private_configuration.certificate_environment["REQUESTS_CA_BUNDLE"] == "/tmp/cert.pem"


def test_private_diagnostic_report_includes_warning(monkeypatch):
    """Private diagnostics should be clearly marked as internal-only."""

    session = _FakeSession()
    monkeypatch.setattr(diagnostics.requests, "Session", lambda: session)
    monkeypatch.setattr(diagnostics, "api_key", lambda: "configured")
    _patch_network_path(monkeypatch)

    report = diagnostics.diagnose_connection(
        api_samples=1,
        verbose=False,
        include_private_network_details=True,
    )

    assert report.privacy_mode == "private"
    assert report.private_details_warning == diagnostics.PRIVATE_DETAILS_WARNING
    assert diagnostics.PRIVATE_DETAILS_WARNING in report.support_text()


# ------ Environment diagnostics ---------------------------------------------------


def test_diagnose_environment_populates_core_fields(monkeypatch):
    """Basic invocation should surface Python, platform, and tidy3d version.

    Fully hermetic: `importlib.metadata.distributions()` and per-package version lookup
    are stubbed so CI never depends on which distributions happen to be installed on
    the runner.
    """

    class _FakeDist:
        def __init__(self, name: str, version: str) -> None:
            self.metadata = {"Name": name}
            self.version = version

    monkeypatch.setattr(
        diagnostics.importlib_metadata,
        "distributions",
        lambda: [_FakeDist("tidy3d", "test-version"), _FakeDist("numpy", "2.0.0")],
    )
    monkeypatch.setattr(
        diagnostics,
        "_package_version",
        lambda name: {"tidy3d": "test-version"}.get(name),
    )

    report = diagnostics.diagnose_environment()

    assert report.tidy3d_version
    assert report.python_version.count(".") == 2
    assert report.python_executable
    assert report.platform
    # `flexcompute_packages` must always list every curated Flexcompute distribution,
    # with ``version=None`` for the ones that aren't installed.
    names_versions = {pkg.name: pkg.version for pkg in report.flexcompute_packages}
    assert names_versions["tidy3d"] == "test-version"
    assert {"tidy3d-extras", "flex-rf", "photonforge", "flow360"}.issubset(names_versions)
    # Only the public `flex-rf` distribution is tracked; the internal `flex_rf` alias must not leak.
    assert "flex_rf" not in names_versions
    # The pip-freeze view is populated from the stubbed distributions list.
    installed_names = {pkg.name for pkg in report.installed_packages}
    assert installed_names == {"tidy3d", "numpy"}
    # Config information must be present so support can see the endpoint / SSL settings
    # without running the connection probe.
    assert report.configuration is not None
    assert report.configuration.api_endpoint


def test_issue_template_and_prompts_stay_consistent():
    """`SUPPORT_REPORT_PROMPTS` must be a strict projection of the seven-slot template.

    Regression guard: if someone appends a prompt to `SUPPORT_REPORT_PROMPTS` directly
    (skipping `_ISSUE_TEMPLATE`) or vice versa, this test fails immediately.
    """

    template = diagnostics._ISSUE_TEMPLATE
    prompts = diagnostics.SUPPORT_REPORT_PROMPTS

    # Every prompted slot in the template appears in SUPPORT_REPORT_PROMPTS, in order.
    template_prompted = tuple(
        (narrative_key, prompt)
        for _question, narrative_key, prompt in template
        if narrative_key is not None and prompt is not None
    )
    assert prompts == template_prompted

    # The template covers the fixed seven-question Tidy3D Issue Report intake form.
    assert len(template) == 7

    # Q5 and Q6 are the two out-of-band slots.
    assert template[4][1] is None and template[4][2] is None
    assert template[5][2] is None


def test_format_issue_template_renders_all_slots():
    """Every slot must render, with "(not provided)" for anything unfilled."""

    env = diagnostics.EnvironmentDiagnosticReport(
        generated_at="G",
        tidy3d_version="1.0.0",
        python_version="3.13.0",
        python_full_version="3.13.0",
        python_executable="/e",
        platform="Linux",
        machine="x86_64",
        processor=None,
        in_virtualenv=True,
        in_notebook=False,
        flexcompute_packages=(),
        installed_packages=(),
    )
    rendered = diagnostics._format_issue_template({}, task_id=None, environment=env)

    header_lines = [line for line in rendered if line and line[0].isdigit()]
    assert len(header_lines) == 7
    text = "\n".join(rendered)
    assert "1. Brief description of the issue:" in text
    assert "4. Tidy3D version and how you run it" in text
    assert "tidy3d 1.0.0" in text
    assert "5. Task ID or task link (if applicable):" in text
    # Every empty slot uses the "(not provided)" placeholder.
    assert text.count("(not provided)") == 6


def test_package_version_returns_none_for_arbitrary_metadata_errors(monkeypatch):
    """`_package_version` must swallow every metadata error so one corrupt dist-info
    cannot abort the whole environment report."""

    def raising(_name):
        raise OSError("corrupt METADATA")

    monkeypatch.setattr(diagnostics.importlib_metadata, "version", raising)
    # Both a `_package_version` call and the wrapping `_collect_packages` iteration must
    # survive the error and report ``version=None``.
    assert diagnostics._package_version("tidy3d") is None
    packages = diagnostics._collect_packages(("tidy3d", "flex-rf"))
    assert [pkg.version for pkg in packages] == [None, None]


def test_installed_distributions_skips_bad_metadata(monkeypatch):
    """One corrupt distribution must not abort the whole package scan."""

    class _GoodDist:
        version = "1.2.3"
        metadata = {"Name": "good-pkg"}

    class _BrokenDist:
        # Raises on metadata access; a known importlib.metadata footgun in the wild.
        @property
        def metadata(self):
            raise ValueError("corrupt METADATA file")

        version = "0"

    class _NamelessDist:
        version = "0"
        metadata = None  # e.g. WHEEL-only dist without Name

    monkeypatch.setattr(
        diagnostics.importlib_metadata,
        "distributions",
        lambda: [_BrokenDist(), _GoodDist(), _NamelessDist()],
    )

    packages = diagnostics._installed_distributions()

    names = [pkg.name for pkg in packages]
    assert names == ["good-pkg"]


def test_installed_distributions_survives_top_level_failure(monkeypatch):
    """If `distributions()` itself raises we return an empty tuple, not a traceback."""

    def raising():
        raise OSError("boom")

    monkeypatch.setattr(diagnostics.importlib_metadata, "distributions", raising)
    assert diagnostics._installed_distributions() == ()


def test_diagnose_environment_marks_private_mode():
    """Private mode must set privacy_mode + surface the warning inline."""

    report = diagnostics.diagnose_environment(include_private_network_details=True)
    assert report.privacy_mode == "private"
    assert report.private_details_warning == diagnostics.PRIVATE_DETAILS_WARNING
    assert diagnostics.PRIVATE_DETAILS_WARNING in report.support_text()


def test_diagnose_environment_default_mode_is_shareable():
    report = diagnostics.diagnose_environment()
    assert report.privacy_mode == "shareable"
    assert report.private_details_warning is None
    assert diagnostics.PRIVATE_DETAILS_WARNING not in report.support_text()


def test_environment_support_text_lists_packages_and_config():
    configuration = diagnostics.ConnectionDiagnosticConfiguration(
        api_endpoint="https://api.example.com",
        ssl_verify=True,
        ssl_version=None,
        proxy_environment={"HTTPS_PROXY": None},
        certificate_environment={"REQUESTS_CA_BUNDLE": None},
        tidy3d_environment={"TIDY3D_WEB__API_ENDPOINT": None},
        warnings=(),
    )
    report = diagnostics.EnvironmentDiagnosticReport(
        generated_at="2026-06-18T00:00:00+00:00",
        tidy3d_version="test",
        python_version="3.13.0",
        python_full_version="3.13.0",
        python_executable="/tmp/py",
        platform="Linux",
        machine="x86_64",
        processor="cpu",
        in_virtualenv=True,
        in_notebook=False,
        flexcompute_packages=(
            diagnostics.EnvironmentPackage(name="tidy3d", version="test"),
            diagnostics.EnvironmentPackage(name="tidy3d-extras", version="1.0"),
            diagnostics.EnvironmentPackage(name="flex-rf", version=None),
        ),
        installed_packages=(
            diagnostics.EnvironmentPackage(name="numpy", version="2.0.0"),
            diagnostics.EnvironmentPackage(name="pydantic", version="2.13.3"),
        ),
        configuration=configuration,
    )
    text = report.support_text()
    assert "Flexcompute packages:" in text
    assert "- tidy3d: test" in text
    assert "- tidy3d-extras: 1.0" in text
    assert "- flex-rf: not installed" in text
    assert "Configuration:" in text
    assert "api_endpoint: https://api.example.com" in text
    assert "ssl_verify: True" in text
    assert "Installed packages (pip freeze):" in text
    assert "numpy==2.0.0" in text
    assert "pydantic==2.13.3" in text


# ------ Combined support report ---------------------------------------------------


def test_diagnose_report_bundles_environment_and_connection(monkeypatch):
    """`diagnose_report` should stitch env + connection into a single SupportReport."""

    def fake_env(**_kwargs):
        return diagnostics.EnvironmentDiagnosticReport(
            generated_at="2026-06-18T00:00:00+00:00",
            tidy3d_version="test",
            python_version="3.13.0",
            python_full_version="3.13.0",
            python_executable="/tmp/py",
            platform="Linux",
            machine="x86_64",
            processor="cpu",
            in_virtualenv=True,
            in_notebook=False,
            flexcompute_packages=(),
            installed_packages=(),
        )

    def fake_conn(**kwargs):
        return diagnostics.ConnectionDiagnosticReport(
            generated_at="2026-06-18T00:00:00+00:00",
            tidy3d_version="test",
            python_version="3.13.0",
            platform="Linux",
            api_endpoint_host="api.example.com",
            api_key_configured=True,
            results=(),
        )

    monkeypatch.setattr(diagnostics, "diagnose_environment", fake_env)
    monkeypatch.setattr(diagnostics, "diagnose_connection", fake_conn)

    report = diagnostics.diagnose_report(
        task_id="task-42",
        narrative={"description": "hello", "run_mode": "Python API", "empty": ""},
    )

    assert report.task_id == "task-42"
    # Empty narrative fields must not appear.
    assert "empty" not in report.narrative
    assert report.narrative == {"description": "hello", "run_mode": "Python API"}
    assert report.connection is not None

    text = report.support_text()
    assert "Tidy3D support report" in text
    # The seven-question issue template is always rendered so support can read the
    # user's answers even if some were left blank.
    assert "Tidy3D Issue Report" in text
    assert "1. Brief description of the issue:" in text
    assert "   hello" in text
    assert "3. Is it reproducible?" in text
    assert "   (not provided)" in text  # unanswered questions still appear
    assert "4. Tidy3D version and how you run it" in text
    assert "   tidy3d test - Python API" in text
    assert "5. Task ID or task link" in text
    assert "   task-42" in text
    assert "7. Could you share a relevant script or model?" in text
    assert "Tidy3D environment" in text
    assert "Tidy3D connection diagnostics" in text


def test_diagnose_report_marks_private_mode(monkeypatch):
    """`diagnose_report(include_private_network_details=True)` must propagate the marker."""

    def fake_env(**kwargs):
        private = kwargs.get("include_private_network_details")
        return diagnostics.EnvironmentDiagnosticReport(
            generated_at="2026-06-18T00:00:00+00:00",
            privacy_mode="private" if private else "shareable",
            private_details_warning=diagnostics.PRIVATE_DETAILS_WARNING if private else None,
            tidy3d_version="test",
            python_version="3.13.0",
            python_full_version="3.13.0",
            python_executable="/tmp/py",
            platform="Linux",
            machine="x86_64",
            processor="cpu",
            in_virtualenv=True,
            in_notebook=False,
            flexcompute_packages=(),
            installed_packages=(),
        )

    monkeypatch.setattr(diagnostics, "diagnose_environment", fake_env)
    monkeypatch.setattr(
        diagnostics,
        "diagnose_connection",
        lambda **_: diagnostics.ConnectionDiagnosticReport(
            generated_at="2026-06-18T00:00:00+00:00",
            tidy3d_version="test",
            python_version="3.13.0",
            platform="Linux",
            api_endpoint_host="api.example.com",
            api_key_configured=True,
            results=(),
        ),
    )

    report = diagnostics.diagnose_report(include_private_network_details=True)
    assert report.privacy_mode == "private"
    assert report.private_details_warning == diagnostics.PRIVATE_DETAILS_WARNING
    text = report.support_text()
    # Warning must surface at the top of the SupportReport, before the environment block.
    warning_pos = text.find(diagnostics.PRIVATE_DETAILS_WARNING)
    env_pos = text.find("Tidy3D environment")
    assert 0 <= warning_pos < env_pos


def test_diagnose_report_can_skip_connection(monkeypatch):
    monkeypatch.setattr(
        diagnostics,
        "diagnose_environment",
        lambda **_: diagnostics.EnvironmentDiagnosticReport(
            generated_at="2026-06-18T00:00:00+00:00",
            tidy3d_version="test",
            python_version="3.13.0",
            python_full_version="3.13.0",
            python_executable="/tmp/py",
            platform="Linux",
            machine="x86_64",
            processor="cpu",
            in_virtualenv=True,
            in_notebook=False,
            flexcompute_packages=(),
            installed_packages=(),
        ),
    )

    def _forbidden(**_kwargs):  # pragma: no cover - guard against accidental call
        raise AssertionError("diagnose_connection must not be called when run_connection=False")

    monkeypatch.setattr(diagnostics, "diagnose_connection", _forbidden)

    report = diagnostics.diagnose_report(run_connection=False)
    assert report.connection is None
    assert "Tidy3D connection diagnostics" not in report.support_text()

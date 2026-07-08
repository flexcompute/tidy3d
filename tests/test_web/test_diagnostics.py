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

"""Connection, environment, and support diagnostics used for support debugging."""

from __future__ import annotations

import os
import platform
import re
import socket
import ssl
import statistics
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field

from tidy3d.config import config
from tidy3d.log import get_logging_console, log
from tidy3d.version import __version__
from tidy3d.web.core.http_util import TLSAdapter, api_key, api_key_auth, ssl_context_for_config

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_TIMEOUT = 20.0
DEFAULT_API_SAMPLES = 3
DEFAULT_DOWNLOAD_BYTES = 100 * 1024 * 1024
DEFAULT_DOWNLOAD_URL_ENDPOINT = "tidy3d/diagnostics/download-url"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_URL_KEYS = ("download_url", "downloadUrl", "url")
PROXY_ENV_VARS = ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy")
CA_ENV_VARS = ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE", "SSL_CERT_DIR")
TIDY3D_ENV_VARS = (
    "TIDY3D_WEB__SSL_VERIFY",
    "TIDY3D_WEB__SSL_VERSION",
    "TIDY3D_WEB__API_ENDPOINT",
    "TIDY3D_SSL_VERIFY",
)
CERTIFICATE_VERIFY_FAILURE_MARKERS = (
    "certificate verify failed",
    "certificate_verify_failed",
    "self-signed certificate",
    "self signed certificate",
    "unable to get local issuer certificate",
)
PRIVATE_DETAILS_WARNING = (
    "PRIVATE NETWORK DETAILS: this report may include internal proxy hosts, NO_PROXY entries, "
    "certificate paths, resolved IP addresses, and TLS certificate details. Share it only with "
    "your institution's IT administrators, not with Flexcompute or anyone outside your institution."
)


class ConnectionDiagnosticSample(BaseModel):
    """One timed connection diagnostic measurement."""

    seconds: float = Field(title="Elapsed wall-clock seconds.")
    bytes_transferred: int | None = Field(None, title="Number of bytes transferred.")
    throughput_mib_s: float | None = Field(None, title="Throughput in MiB/s.")


class ConnectionDiagnosticResult(BaseModel):
    """Result for one connection diagnostic check."""

    name: str = Field(title="Diagnostic check name.")
    status: str = Field(title="One of 'pass', 'fail', or 'skip'.")
    target_host: str | None = Field(None, title="Host tested, without URL secrets.")
    samples: tuple[ConnectionDiagnosticSample, ...] = Field(
        (), title="Timed measurements for this check."
    )
    detail: str | None = Field(None, title="Human-readable summary.")
    error_type: str | None = Field(None, title="Exception type if the check failed.")
    error: str | None = Field(None, title="Sanitized exception message if the check failed.")
    recommendation: str | None = Field(None, title="Suggested next step for known failures.")
    metadata: dict[str, Any] = Field(default_factory=dict, title="Structured diagnostic metadata.")


class ConnectionDiagnosticConfiguration(BaseModel):
    """Redacted client network configuration included with diagnostic reports."""

    api_endpoint: str = Field(title="Configured API endpoint.")
    ssl_verify: bool = Field(title="Configured SSL certificate verification setting.")
    ssl_version: str | None = Field(None, title="Configured TLS version override.")
    proxy_environment: dict[str, str | None] = Field(
        title="Redacted proxy-related environment variables."
    )
    certificate_environment: dict[str, str | None] = Field(
        title="Redacted certificate-related environment variables."
    )
    tidy3d_environment: dict[str, str | None] = Field(
        title="Redacted Tidy3D network environment variables."
    )
    warnings: tuple[str, ...] = Field((), title="Configuration warnings.")


class ConnectionDiagnosticReport(BaseModel):
    """Support-facing network diagnostic report."""

    generated_at: str = Field(title="UTC report creation timestamp.")
    privacy_mode: str = Field("shareable", title="Either 'shareable' or 'private'.")
    private_details_warning: str | None = Field(
        None, title="Warning shown when private network details are included."
    )
    tidy3d_version: str = Field(title="Tidy3D client version.")
    python_version: str = Field(title="Python version.")
    platform: str = Field(title="Operating system and machine summary.")
    api_endpoint_host: str | None = Field(None, title="Configured API endpoint host.")
    api_key_configured: bool = Field(title="Whether an API key is configured.")
    configuration: ConnectionDiagnosticConfiguration | None = Field(
        None, title="Redacted client network configuration."
    )
    results: tuple[ConnectionDiagnosticResult, ...] = Field(title="Diagnostic check results.")

    def support_text(self) -> str:
        """Return a paste-friendly support summary with structured JSON."""

        lines = ["Tidy3D connection diagnostics"]
        lines.extend(self._body_lines(include_privacy_header=True, include_configuration=True))
        lines.extend(["", "JSON:", self.model_dump_json(indent=2)])
        return "\n".join(lines)

    def _body_lines(
        self, *, include_privacy_header: bool, include_configuration: bool
    ) -> list[str]:
        """Return the report body without the outer heading or JSON dump.

        ``include_privacy_header`` and ``include_configuration`` let
        :class:`SupportReport` compose the combined bundle without duplicating
        the privacy warning or the redacted configuration snapshot that its
        environment section already carries.
        """

        lines = [f"- generated_at: {self.generated_at}"]
        if include_privacy_header:
            lines.append(f"- privacy_mode: {self.privacy_mode}")
        lines.extend(
            [
                f"- tidy3d_version: {self.tidy3d_version}",
                f"- python_version: {self.python_version}",
                f"- platform: {self.platform}",
                f"- api_endpoint_host: {self.api_endpoint_host or 'unknown'}",
                f"- api_key_configured: {self.api_key_configured}",
            ]
        )
        if include_privacy_header and self.private_details_warning:
            lines.extend(["", self.private_details_warning, ""])
        if include_configuration and self.configuration is not None:
            # Shared with the environment report so both surfaces render the same
            # redacted configuration fields in the same format.
            lines.extend(_format_configuration_lines(self.configuration))

        lines.extend(["", "Checks:"])
        for result in self.results:
            summary = _result_summary(result)
            lines.append(f"- {result.name}: {result.status}{summary}")
            if result.recommendation:
                lines.append(f"  recommendation: {result.recommendation}")
        return lines


def _host_from_url(url: str | None) -> str | None:
    """Return only the host from a URL so signed URL secrets are not exposed."""

    if not url:
        return None
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        return None
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        return f"{host}:{parsed.port}"
    return host


def _redact_url(url: str) -> str:
    """Strip query and fragment data from a URL."""

    parsed = urlparse(url)
    if not (parsed.scheme and parsed.netloc) and not parsed.path.startswith("/"):
        return url
    return parsed._replace(query="", fragment="").geturl()


def _redact_urls_in_text(value: str) -> str:
    """Strip query and fragment data from any URLs in text."""

    return re.sub(r"(?:https?://|/)[^\s)>\]\"']+", lambda match: _redact_url(match.group(0)), value)


def _redact_env_value(value: str) -> str:
    """Redact credentials and URL secrets from environment values."""

    parsed = urlparse(value)
    if not (parsed.scheme and parsed.netloc):
        return _redact_urls_in_text(value)

    netloc = _host_from_url(value) or ""
    return parsed._replace(netloc=netloc, query="", fragment="").geturl()


def _redacted_env(names: tuple[str, ...]) -> dict[str, str | None]:
    """Return selected environment variables with sensitive URL parts removed."""

    return {
        name: _redact_env_value(os.environ[name]) if name in os.environ else None for name in names
    }


def _replace_if_present(message: str, value: str, replacement: str) -> str:
    """Replace a non-empty value in ``message``."""

    if not value:
        return message
    return message.replace(value, replacement)


def _sanitize_public_error_message(message: str) -> str:
    """Remove private env-derived network details from support-shareable errors."""

    redacted = message
    for name in PROXY_ENV_VARS:
        value = os.environ.get(name)
        if not value:
            continue
        redacted = _replace_if_present(redacted, value, "<redacted_proxy_env>")
        parsed = urlparse(value)
        if parsed.netloc:
            redacted = _replace_if_present(redacted, parsed.netloc, "<redacted_proxy>")
        if parsed.hostname:
            redacted = _replace_if_present(redacted, parsed.hostname, "<redacted_proxy>")

    for name in CA_ENV_VARS:
        value = os.environ.get(name)
        if value:
            redacted = _replace_if_present(redacted, value, "<redacted_certificate_path>")

    return redacted


def _env_presence(names: tuple[str, ...]) -> dict[str, str | None]:
    """Return whether selected environment variables are set without exposing their values."""

    return {name: "set" if name in os.environ else None for name in names}


def _format_env_summary(values: dict[str, str | None]) -> str:
    """Return a compact support-text summary for environment variables."""

    set_names = [name for name, value in values.items() if value is not None]
    if not set_names:
        return "none set"
    return ", ".join(f"{name}={values[name]}" for name in set_names)


def _diagnostic_configuration(
    *, include_private_network_details: bool = False
) -> ConnectionDiagnosticConfiguration:
    """Build a redacted configuration snapshot for support reports."""

    tidy3d_environment = _redacted_env(TIDY3D_ENV_VARS)
    warnings = []
    if tidy3d_environment.get("TIDY3D_SSL_VERIFY") is not None:
        warnings.append(
            "TIDY3D_SSL_VERIFY is set but ignored; use TIDY3D_WEB__SSL_VERIFY to configure "
            "SSL verification."
        )

    return ConnectionDiagnosticConfiguration(
        api_endpoint=_redact_env_value(str(config.web.api_endpoint)),
        ssl_verify=bool(config.web.ssl_verify),
        ssl_version=config.web.ssl_version,
        proxy_environment=(
            _redacted_env(PROXY_ENV_VARS)
            if include_private_network_details
            else _env_presence(PROXY_ENV_VARS)
        ),
        certificate_environment=(
            _redacted_env(CA_ENV_VARS)
            if include_private_network_details
            else _env_presence(CA_ENV_VARS)
        ),
        tidy3d_environment=tidy3d_environment,
        warnings=tuple(warnings),
    )


def _sanitize_error_message(message: str, *, include_private_network_details: bool = False) -> str:
    """Remove signed URL query strings and optionally private network details from error text."""

    redacted = _redact_urls_in_text(message)
    if include_private_network_details:
        return redacted
    return _sanitize_public_error_message(redacted)


def _exception_chain(exc: Exception) -> tuple[BaseException, ...]:
    """Return the explicit and implicit exception chain for classification."""

    chain: list[BaseException] = []
    current: BaseException | None = exc
    seen = set()
    while current is not None and id(current) not in seen:
        chain.append(current)
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _is_certificate_verify_failure(exc: Exception) -> bool:
    """Return whether an exception looks like Python CA verification failure."""

    for chained in _exception_chain(exc):
        if isinstance(chained, ssl.SSLCertVerificationError):
            return True
        message = str(chained).lower()
        if any(marker in message for marker in CERTIFICATE_VERIFY_FAILURE_MARKERS):
            return True
    return False


def _recommendation_for_exception(exc: Exception, target_url: str | None) -> str | None:
    """Return a next-step recommendation for a known connectivity failure."""

    if not _is_certificate_verify_failure(exc):
        return None
    target_host = _host_from_url(target_url) or "the configured Tidy3D endpoint"
    return (
        f"Python certificate verification failed for {target_host}. This often happens on "
        "managed networks that inspect HTTPS traffic or use private CA certificates. If "
        "the network CA is already trusted by the operating system/browser, run "
        "`python -m pip install --trusted-host pypi.org --trusted-host "
        "files.pythonhosted.org pip-system-certs` in the same Python environment, then "
        "rerun diagnostics. Otherwise set REQUESTS_CA_BUNDLE or CURL_CA_BUNDLE to the "
        f"network CA bundle, or ask IT to allow access to {target_host}."
    )


def _error_result(
    name: str,
    target_url: str | None,
    exc: Exception,
    *,
    include_private_network_details: bool = False,
) -> ConnectionDiagnosticResult:
    """Build a failed result without including signed URL query strings."""

    return ConnectionDiagnosticResult(
        name=name,
        status="fail",
        target_host=_host_from_url(target_url),
        error_type=type(exc).__name__,
        error=_sanitize_error_message(
            str(exc), include_private_network_details=include_private_network_details
        ),
        recommendation=_recommendation_for_exception(exc, target_url),
    )


def _configured_session() -> requests.Session:
    """Create a requests session using the same SSL knobs as the main client."""

    session = requests.Session()
    session.verify = config.web.ssl_verify
    if config.web.ssl_version and hasattr(session, "mount"):
        session.mount("https://", TLSAdapter())
    return session


def _requests_ca_bundle_locations() -> tuple[str | None, str | None]:
    """Return cafile/capath settings using Requests' CA bundle precedence."""

    ca_bundle = (
        os.environ.get("REQUESTS_CA_BUNDLE")
        or os.environ.get("CURL_CA_BUNDLE")
        or requests.certs.where()
    )
    if not ca_bundle:
        return None, None
    if os.path.isdir(ca_bundle):
        return None, ca_bundle
    return ca_bundle, None


def _ssl_context_for_config() -> ssl.SSLContext:
    """Create a TLS context that mirrors the configured verification behavior."""

    context = ssl_context_for_config(
        cert_reqs=ssl.CERT_REQUIRED if config.web.ssl_verify else ssl.CERT_NONE,
    )
    if config.web.ssl_verify:
        cafile, capath = _requests_ca_bundle_locations()
        if cafile or capath:
            context.load_verify_locations(cafile=cafile, capath=capath)
    return context


def _certificate_name(name: tuple[tuple[tuple[str, str], ...], ...] | None) -> str | None:
    """Convert a certificate subject/issuer tuple into a compact string."""

    if not name:
        return None
    parts = []
    for attributes in name:
        for key, value in attributes:
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else None


def _connect_to_resolved_address(addr_info: list[tuple[Any, ...]], timeout: float) -> socket.socket:
    """Open a socket using addresses already returned by getaddrinfo."""

    last_error: OSError | None = None
    for family, socktype, proto, _, sockaddr in addr_info:
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        try:
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            last_error = exc
            sock.close()
    if last_error is not None:
        raise last_error
    raise OSError("No resolved addresses to connect to.")


def _check_network_path(
    *, api_url: str, timeout: float, include_private_network_details: bool = False
) -> ConnectionDiagnosticResult:
    """Check DNS resolution, TCP connectivity, and TLS handshake to the API host."""

    parsed = urlparse(api_url)
    host = parsed.hostname
    if not host:
        return _error_result(
            "network_path",
            api_url,
            ValueError("API endpoint has no host."),
            include_private_network_details=include_private_network_details,
        )
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    try:
        dns_start = time.perf_counter()
        addr_info = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        dns_seconds = max(time.perf_counter() - dns_start, sys.float_info.epsilon)
        resolved_ips = tuple(dict.fromkeys(info[4][0] for info in addr_info))
        if not resolved_ips:
            raise OSError(f"No addresses resolved for {host}.")

        tcp_start = time.perf_counter()
        with _connect_to_resolved_address(addr_info, timeout=timeout) as sock:
            tcp_seconds = max(time.perf_counter() - tcp_start, sys.float_info.epsilon)
            remote_ip = sock.getpeername()[0] if include_private_network_details else None
            metadata: dict[str, Any] = {
                "mode": "direct_socket",
                "dns_seconds": dns_seconds,
                "tcp_seconds": tcp_seconds,
                "resolved_ip_count": len(resolved_ips),
                "port": port,
            }

            if parsed.scheme == "https":
                tls_start = time.perf_counter()
                with _ssl_context_for_config().wrap_socket(sock, server_hostname=host) as tls_sock:
                    metadata.update(
                        {
                            "tls_seconds": max(
                                time.perf_counter() - tls_start, sys.float_info.epsilon
                            ),
                            "tls_version": tls_sock.version(),
                            "tls_cipher": tls_sock.cipher()[0] if tls_sock.cipher() else None,
                        }
                    )
                    try:
                        certificate_present = bool(tls_sock.getpeercert(binary_form=True))
                    except TypeError:
                        certificate_present = bool(tls_sock.getpeercert())
                    metadata["certificate_present"] = certificate_present
                    if include_private_network_details:
                        metadata["resolved_ips"] = resolved_ips
                        metadata["remote_ip"] = remote_ip
                        certificate = tls_sock.getpeercert()
                        if certificate:
                            metadata.update(
                                {
                                    "certificate_subject": _certificate_name(
                                        certificate.get("subject")
                                    ),
                                    "certificate_issuer": _certificate_name(
                                        certificate.get("issuer")
                                    ),
                                    "certificate_not_after": certificate.get("notAfter"),
                                }
                            )

        detail_parts = [
            f"direct dns {dns_seconds * 1000:.1f} ms",
            f"tcp {tcp_seconds * 1000:.1f} ms",
        ]
        if "tls_version" in metadata:
            detail_parts.append(f"tls {metadata['tls_version']}")
        return ConnectionDiagnosticResult(
            name="network_path",
            status="pass",
            target_host=_host_from_url(api_url),
            detail=", ".join(detail_parts),
            metadata=metadata,
        )
    except Exception as exc:
        result = _error_result(
            "network_path",
            api_url,
            exc,
            include_private_network_details=include_private_network_details,
        )
        result.metadata["mode"] = "direct_socket"
        return result


def _timed_sample(
    action: Callable[[], int | None],
) -> ConnectionDiagnosticSample:
    """Run ``action`` and return a timing sample."""

    start = time.perf_counter()
    bytes_transferred = action()
    seconds = max(time.perf_counter() - start, sys.float_info.epsilon)
    throughput = None
    if bytes_transferred is not None:
        throughput = bytes_transferred / 1024 / 1024 / seconds
    return ConnectionDiagnosticSample(
        seconds=seconds,
        bytes_transferred=bytes_transferred,
        throughput_mib_s=throughput,
    )


def _result_summary(result: ConnectionDiagnosticResult) -> str:
    """Return a compact text summary for one diagnostic result."""

    if result.detail:
        return f" ({result.detail})"
    if result.error:
        return f" ({result.error_type}: {result.error})"
    if not result.samples:
        return ""

    seconds = [sample.seconds for sample in result.samples]
    latency_ms = statistics.median(seconds) * 1000
    throughputs = [
        sample.throughput_mib_s for sample in result.samples if sample.throughput_mib_s is not None
    ]
    if throughputs:
        return f" (median {statistics.median(throughputs):.2f} MiB/s)"
    return f" (median {latency_ms:.1f} ms)"


def _response_json(response: requests.Response, endpoint_name: str) -> Any:
    """Return response JSON or fail without including potentially private body text."""

    try:
        return response.json()
    except ValueError as exc:
        headers = getattr(response, "headers", {}) or {}
        content_type = headers.get("content-type") or headers.get("Content-Type") or "unknown"
        content_type = str(content_type).replace("/", "_")
        raise ValueError(
            f"{endpoint_name} endpoint returned a non-JSON response "
            f"(content-type: {content_type}; parse error: {exc})."
        ) from exc


def _check_health_response(response: requests.Response) -> None:
    """Validate the health endpoint body so proxy block pages do not count as success."""

    payload = _response_json(response, "Health")
    if not isinstance(payload, dict) or payload.get("health") != "OK":
        raise ValueError("Health endpoint returned unexpected JSON payload.")


def _check_authentication_response(response: requests.Response) -> None:
    """Validate the projects endpoint body so arbitrary JSON does not count as auth success."""

    payload = _response_json(response, "Authentication")
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Authentication endpoint returned unexpected JSON payload.")


def _check_api_latency(
    *,
    session: requests.Session,
    api_url: str,
    samples: int,
    timeout: float,
    include_private_network_details: bool = False,
) -> ConnectionDiagnosticResult:
    """Measure latency to a lightweight API endpoint."""

    try:
        measurements = []
        for _ in range(samples):

            def request_once() -> None:
                response = session.get(api_url, timeout=timeout)
                if not response.ok:
                    response.raise_for_status()
                _check_health_response(response)
                return None

            measurements.append(_timed_sample(request_once))
        return ConnectionDiagnosticResult(
            name="api_latency",
            status="pass",
            target_host=_host_from_url(api_url),
            samples=tuple(measurements),
        )
    except Exception as exc:
        return _error_result(
            "api_latency",
            api_url,
            exc,
            include_private_network_details=include_private_network_details,
        )


def _check_authentication(
    *,
    session: requests.Session,
    api_url: str,
    timeout: float,
    include_private_network_details: bool = False,
) -> ConnectionDiagnosticResult:
    """Check whether the configured API key can authenticate against the API."""

    try:

        def request_once() -> None:
            response = session.get(api_url, auth=api_key_auth, timeout=timeout)
            if not response.ok:
                response.raise_for_status()
            _check_authentication_response(response)
            return None

        sample = _timed_sample(request_once)
        return ConnectionDiagnosticResult(
            name="authentication",
            status="pass",
            target_host=_host_from_url(api_url),
            samples=(sample,),
        )
    except Exception as exc:
        return _error_result(
            "authentication",
            api_url,
            exc,
            include_private_network_details=include_private_network_details,
        )


def _read_response_bytes(response: requests.Response, max_bytes: int) -> int:
    """Read up to ``max_bytes`` from a streaming response."""

    bytes_read = 0
    for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
        if not chunk:
            continue
        bytes_read += min(len(chunk), max_bytes - bytes_read)
        if bytes_read >= max_bytes:
            break
    return bytes_read


def _read_expected_response_bytes(response: requests.Response, expected_bytes: int) -> int:
    """Read a streaming response and fail if the requested byte count was not received."""

    bytes_read = _read_response_bytes(response, max_bytes=expected_bytes)
    if bytes_read != expected_bytes:
        raise ValueError(
            "Diagnostic storage download returned "
            f"{bytes_read} bytes, expected {expected_bytes} bytes."
        )
    return bytes_read


def _download_url_from_response(response: requests.Response) -> str:
    """Extract a diagnostic download URL from the backend response."""

    if not response.ok:
        response.raise_for_status()

    payload = response.json()
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in DOWNLOAD_URL_KEYS:
            value = payload.get(key)
            if value:
                return str(value)

    raise ValueError(
        "Diagnostic download URL response must be a URL string or include one of "
        f"{DOWNLOAD_URL_KEYS}."
    )


def _resolve_download_url(
    *,
    session: requests.Session,
    endpoint_url: str,
    timeout: float,
) -> str:
    """Resolve the default diagnostic storage URL from the backend."""

    response = session.get(endpoint_url, auth=api_key_auth, timeout=timeout)
    return _download_url_from_response(response)


def _check_download_throughput(
    *,
    session: requests.Session,
    endpoint_url: str,
    timeout: float,
    include_private_network_details: bool = False,
) -> ConnectionDiagnosticResult:
    """Measure download throughput from the default diagnostic storage object."""

    download_url = endpoint_url
    try:
        download_url = _resolve_download_url(
            session=session,
            endpoint_url=endpoint_url,
            timeout=timeout,
        )
        measurements = []
        final_url = download_url
        headers = {"Range": f"bytes=0-{DEFAULT_DOWNLOAD_BYTES - 1}"}

        def request_once() -> int:
            nonlocal final_url
            response = session.get(
                download_url,
                headers=headers,
                stream=True,
                timeout=timeout,
            )
            try:
                final_url = getattr(response, "url", download_url) or download_url
                if not response.ok:
                    response.raise_for_status()
                return _read_expected_response_bytes(
                    response,
                    expected_bytes=DEFAULT_DOWNLOAD_BYTES,
                )
            finally:
                response.close()

        measurements.append(_timed_sample(request_once))
        return ConnectionDiagnosticResult(
            name="storage_download",
            status="pass",
            target_host=_host_from_url(final_url),
            samples=tuple(measurements),
        )
    except Exception as exc:
        return _error_result(
            "storage_download",
            download_url,
            exc,
            include_private_network_details=include_private_network_details,
        )


def diagnose_connection(
    *,
    api_samples: int = DEFAULT_API_SAMPLES,
    timeout: float = DEFAULT_TIMEOUT,
    verbose: bool = True,
    include_private_network_details: bool = False,
) -> ConnectionDiagnosticReport:
    """Run a support-facing network diagnostic report.

    Parameters
    ----------
    api_samples:
        Number of API latency samples to collect.
    timeout:
        Per-request timeout in seconds.
    verbose:
        If ``True``, print a paste-friendly summary to the Tidy3D console.
    include_private_network_details:
        If ``True``, include internal network details intended only for the user's IT
        administrators. The default ``False`` produces support-shareable output.

    Returns
    -------
    ConnectionDiagnosticReport
        Structured diagnostic report suitable for support tickets.
    """

    if api_samples < 1:
        raise ValueError("'api_samples' must be at least 1.")
    if timeout <= 0:
        raise ValueError("'timeout' must be positive.")

    session = _configured_session()
    health_url = config.web.build_api_url("health")
    auth_url = config.web.build_api_url("tidy3d/projects")
    download_url_endpoint = config.web.build_api_url(DEFAULT_DOWNLOAD_URL_ENDPOINT)
    report = ConnectionDiagnosticReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        privacy_mode="private" if include_private_network_details else "shareable",
        private_details_warning=PRIVATE_DETAILS_WARNING
        if include_private_network_details
        else None,
        tidy3d_version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        api_endpoint_host=_host_from_url(str(config.web.api_endpoint)),
        api_key_configured=api_key() is not None,
        configuration=_diagnostic_configuration(
            include_private_network_details=include_private_network_details
        ),
        results=(
            _check_network_path(
                api_url=str(config.web.api_endpoint),
                timeout=timeout,
                include_private_network_details=include_private_network_details,
            ),
            _check_api_latency(
                session=session,
                api_url=health_url,
                samples=api_samples,
                timeout=timeout,
                include_private_network_details=include_private_network_details,
            ),
            _check_authentication(
                session=session,
                api_url=auth_url,
                timeout=timeout,
                include_private_network_details=include_private_network_details,
            ),
            _check_download_throughput(
                session=session,
                endpoint_url=download_url_endpoint,
                timeout=timeout,
                include_private_network_details=include_private_network_details,
            ),
        ),
    )
    if verbose:
        get_logging_console().print(report.support_text())
    return report


# Packages worth capturing for support tickets. Absence is signalled with ``version=None`` so
# support engineers can distinguish "not installed" from "unknown". Only the public distribution
# names are listed; ``flex-rf`` is the public name for the RF client.
FLEXCOMPUTE_PACKAGES = (
    "tidy3d",
    "tidy3d-extras",
    "flex-rf",
    "photonforge",
    "flow360",
)


class EnvironmentPackage(BaseModel):
    """One installed (or missing) Python package."""

    name: str = Field(title="Distribution name as queried.")
    version: str | None = Field(None, title="Installed version, or None if not importable.")


class EnvironmentDiagnosticReport(BaseModel):
    """Support-facing snapshot of the local Python environment."""

    generated_at: str = Field(title="UTC report creation timestamp.")
    privacy_mode: str = Field("shareable", title="Either 'shareable' or 'private'.")
    private_details_warning: str | None = Field(
        None, title="Warning shown when private network details are embedded."
    )
    tidy3d_version: str = Field(title="Tidy3D client version.")
    python_version: str = Field(title="Python interpreter version (major.minor.patch).")
    python_full_version: str = Field(title="Full Python version string from sys.version.")
    python_executable: str = Field(title="Path to the running Python interpreter.")
    platform: str = Field(title="Operating system and machine summary.")
    machine: str = Field(title="Machine architecture (e.g. x86_64, arm64).")
    processor: str | None = Field(None, title="Processor identifier if reported by the OS.")
    in_virtualenv: bool = Field(title="Whether the interpreter is inside a venv/uv env.")
    in_notebook: bool = Field(
        title="Whether execution appears to be inside a notebook kernel (Jupyter, Colab, VSCode)."
    )
    flexcompute_packages: tuple[EnvironmentPackage, ...] = Field(
        title="Tidy3D and adjacent Flexcompute distributions."
    )
    installed_packages: tuple[EnvironmentPackage, ...] = Field(
        title="All distributions visible via importlib.metadata (pip-freeze equivalent)."
    )
    configuration: ConnectionDiagnosticConfiguration | None = Field(
        None, title="Redacted Tidy3D client configuration snapshot."
    )

    def support_text(self) -> str:
        """Return a paste-friendly summary of the local environment."""

        return "\n".join(["Tidy3D environment", *self._body_lines(include_privacy_header=True)])

    def _body_lines(self, *, include_privacy_header: bool) -> list[str]:
        """Return the environment body without the outer heading.

        ``include_privacy_header`` lets :class:`SupportReport` compose the
        combined bundle without repeating the privacy warning it already
        prints at the top of the report.
        """

        lines = [f"- generated_at: {self.generated_at}"]
        if include_privacy_header:
            lines.append(f"- privacy_mode: {self.privacy_mode}")
        if include_privacy_header and self.private_details_warning:
            lines.extend(["", self.private_details_warning, ""])
        lines.extend(
            [
                f"- tidy3d_version: {self.tidy3d_version}",
                f"- python_version: {self.python_version}",
                f"- python_executable: {self.python_executable}",
                f"- platform: {self.platform}",
                f"- machine: {self.machine}",
                f"- processor: {self.processor or 'unknown'}",
                f"- in_virtualenv: {self.in_virtualenv}",
                f"- in_notebook: {self.in_notebook}",
                "",
                "Flexcompute packages:",
            ]
        )
        lines.extend(_format_package_lines(self.flexcompute_packages))
        if self.configuration is not None:
            lines.extend(["", "Configuration:", *_format_configuration_lines(self.configuration)])
        lines.extend(["", "Installed packages (pip freeze):"])
        lines.extend(_format_pip_freeze_lines(self.installed_packages))
        return lines


class SupportReport(BaseModel):
    """Combined support bundle mapping onto the Tidy3D issue report template."""

    generated_at: str = Field(title="UTC report creation timestamp.")
    privacy_mode: str = Field("shareable", title="Either 'shareable' or 'private'.")
    private_details_warning: str | None = Field(
        None, title="Warning shown when private network details are embedded."
    )
    task_id: str | None = Field(None, title="Task ID the user is reporting on, if any.")
    narrative: dict[str, str] = Field(
        default_factory=dict,
        title="Free-text answers to the support template (description, when, reproducibility).",
    )
    environment: EnvironmentDiagnosticReport = Field(title="Local environment snapshot.")
    connection: ConnectionDiagnosticReport | None = Field(
        None, title="Optional connection diagnostics."
    )

    def support_text(self) -> str:
        """Return a paste-friendly issue-report block."""

        lines = [
            "Tidy3D support report",
            f"- generated_at: {self.generated_at}",
            f"- privacy_mode: {self.privacy_mode}",
        ]
        if self.private_details_warning:
            lines.extend(["", self.private_details_warning, ""])
        lines.extend(
            [
                "",
                "Tidy3D Issue Report",
                "Please fill out what you can - the more detail, the faster we can help.",
                "",
            ]
        )
        lines.extend(_format_issue_template(self.narrative, self.task_id, self.environment))
        # Nested reports skip their own privacy header and the redacted configuration; the
        # combined bundle already prints those once at the top / in the environment section.
        lines.extend(
            [
                "",
                "Tidy3D environment",
                *self.environment._body_lines(include_privacy_header=False),
            ]
        )
        if self.connection is not None:
            lines.extend(
                [
                    "",
                    "Tidy3D connection diagnostics",
                    *self.connection._body_lines(
                        include_privacy_header=False, include_configuration=False
                    ),
                ]
            )
        return "\n".join(lines)


def _format_package_lines(packages: tuple[EnvironmentPackage, ...]) -> list[str]:
    return [
        f"- {package.name}: {package.version if package.version else 'not installed'}"
        for package in packages
    ]


def _format_pip_freeze_lines(packages: tuple[EnvironmentPackage, ...]) -> list[str]:
    """Return one ``name==version`` line per package, matching ``pip freeze`` output."""

    return [
        f"{package.name}=={package.version}" if package.version else package.name
        for package in packages
    ]


def _installed_distributions() -> tuple[EnvironmentPackage, ...]:
    """Enumerate every installed distribution via importlib.metadata, sorted by name.

    Duplicate distributions (same name across multiple sys.path entries) keep the first
    occurrence. Distributions with missing / corrupt metadata are skipped so a single
    broken egg-info cannot abort the whole troubleshoot report — which would defeat the
    point of a tool meant to run in broken environments.
    """

    try:
        distributions = list(importlib_metadata.distributions())
    except Exception as exc:
        log.debug(f"importlib.metadata.distributions() failed: {exc!r}")
        return ()

    entries: dict[str, EnvironmentPackage] = {}
    for dist in distributions:
        try:
            metadata = dist.metadata
            name = metadata["Name"] if metadata else None
            version = dist.version
        except Exception as exc:
            log.debug(f"skipping distribution with unreadable metadata: {exc!r}")
            continue
        if not name:
            continue
        canonical = name.replace("_", "-").lower()
        if canonical in entries:
            continue
        entries[canonical] = EnvironmentPackage(name=name, version=version)
    return tuple(entries[key] for key in sorted(entries))


def _format_configuration_lines(configuration: ConnectionDiagnosticConfiguration) -> list[str]:
    """Return a paste-friendly bullet list of the redacted client configuration."""

    lines = [
        f"- api_endpoint: {configuration.api_endpoint}",
        f"- ssl_verify: {configuration.ssl_verify}",
        f"- ssl_version: {configuration.ssl_version or 'default'}",
        f"- proxy_env: {_format_env_summary(configuration.proxy_environment)}",
        f"- certificate_env: {_format_env_summary(configuration.certificate_environment)}",
        f"- tidy3d_env: {_format_env_summary(configuration.tidy3d_environment)}",
    ]
    lines.extend(f"- warning: {warning}" for warning in configuration.warnings)
    return lines


def _package_version(name: str) -> str | None:
    """Return the installed version of ``name`` if visible via importlib.metadata.

    Any error (missing package, corrupt METADATA, unreadable dist-info) resolves to
    ``None`` so a single broken adjacent distribution cannot abort a report meant to
    run in exactly that kind of environment.
    """

    try:
        return importlib_metadata.version(name)
    except Exception as exc:
        log.debug(f"failed to read installed version for {name!r}: {exc!r}")
        return None


def _collect_packages(names: tuple[str, ...]) -> tuple[EnvironmentPackage, ...]:
    return tuple(EnvironmentPackage(name=name, version=_package_version(name)) for name in names)


def _in_virtualenv() -> bool:
    """Return True when the current interpreter is inside a venv / uv-managed env."""

    base_prefix = getattr(sys, "base_prefix", sys.prefix)
    return sys.prefix != base_prefix


def _running_in_notebook() -> bool:
    """Return True when running inside a notebook kernel (Jupyter, Colab, VSCode, ...).

    The check is delegated to `tidy3d.components.viz.axes_utils._is_notebook` — the same
    helper the rest of the client uses. Imported lazily so the CLI (in particular
    ``tidy3d troubleshoot``) does not pay the cost of loading ``tidy3d.components.viz``
    on startup for non-notebook use.
    """

    try:
        from tidy3d.components.viz.axes_utils import _is_notebook
    except Exception as exc:
        log.debug(f"notebook detection unavailable: {exc!r}")
        return False
    return _is_notebook()


def diagnose_environment(
    *,
    verbose: bool = False,
    include_private_network_details: bool = False,
) -> EnvironmentDiagnosticReport:
    """Collect a support-facing snapshot of the local Python environment.

    Parameters
    ----------
    verbose:
        If ``True``, echo the report to the Tidy3D console.
    include_private_network_details:
        If ``True``, embed proxy hosts and certificate paths in the configuration snapshot.
        The default ``False`` produces support-shareable output.

    Returns
    -------
    EnvironmentDiagnosticReport
        Structured environment snapshot suitable for a support ticket.
    """

    flexcompute_packages = _collect_packages(FLEXCOMPUTE_PACKAGES)
    installed_packages = _installed_distributions()
    report = EnvironmentDiagnosticReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        tidy3d_version=__version__,
        python_version=platform.python_version(),
        python_full_version=sys.version.replace("\n", " "),
        python_executable=sys.executable,
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor() or None,
        in_virtualenv=_in_virtualenv(),
        in_notebook=_running_in_notebook(),
        flexcompute_packages=flexcompute_packages,
        installed_packages=installed_packages,
        configuration=_diagnostic_configuration(
            include_private_network_details=include_private_network_details
        ),
        privacy_mode="private" if include_private_network_details else "shareable",
        private_details_warning=PRIVATE_DETAILS_WARNING
        if include_private_network_details
        else None,
    )
    if verbose:
        get_logging_console().print(report.support_text())
    return report


# Ordered, single source of truth for the seven-question Tidy3D Issue Report template.
# Each slot describes:
#   - ``question``: the exact wording rendered in the report.
#   - ``narrative_key``: dict key in the ``narrative`` mapping, or ``None`` if the answer
#     is supplied out-of-band (Q5 from ``--task-id``, Q4 auto-fills the tidy3d version).
#   - ``prompt``: interactive prompt shown by ``tidy3d troubleshoot report``, or ``None``
#     if the answer isn't collected interactively (Q5 uses ``--task-id``; Q6 uses
#     ``--traceback-file`` for multi-line content).
# Add a new question by appending one tuple here; no other file should need to change.
_IssueSlot = tuple[str, str | None, str | None]

_ISSUE_TEMPLATE: tuple[_IssueSlot, ...] = (
    (
        "Brief description of the issue",
        "description",
        "Brief description of the issue",
    ),
    (
        "When did you first encounter it? If this worked before, what changed "
        "(version upgrade, new environment, modified model)?",
        "when_started",
        "When did you first encounter it? If this worked before, what changed "
        "(version upgrade, new environment, modified model)?",
    ),
    (
        "Is it reproducible? (every time / intermittent / happened once)",
        "reproducibility",
        "Is it reproducible? (every time / intermittent / happened once)",
    ),
    (
        "Tidy3D version and how you run it (Python API / web GUI / notebook)",
        "run_mode",
        "How are you running Tidy3D? (Python API / web GUI / notebook)",
    ),
    (
        "Task ID or task link (if applicable)",
        None,
        None,
    ),
    (
        "Steps to reproduce, with the full error message/traceback pasted as text",
        "steps_to_reproduce",
        None,
    ),
    (
        "Could you share a relevant script or model?",
        "script_share",
        "Could you share a relevant script or model? (path / gist / 'yes, will attach' / 'no')",
    ),
)


# Public alias used by the CLI to drive the interactive prompt loop. Derived from
# ``_ISSUE_TEMPLATE`` so a new prompt is added by extending the template, not by
# keeping two parallel structures in sync.
SUPPORT_REPORT_PROMPTS: tuple[tuple[str, str], ...] = tuple(
    (narrative_key, prompt)
    for _question, narrative_key, prompt in _ISSUE_TEMPLATE
    if narrative_key is not None and prompt is not None
)


def _format_issue_template(
    narrative: dict[str, str],
    task_id: str | None,
    environment: EnvironmentDiagnosticReport,
) -> list[str]:
    """Render the Tidy3D Issue Report template with the user's answers inlined.

    Every slot in :data:`_ISSUE_TEMPLATE` renders — questions with no answer print
    ``(not provided)`` so the printed report always shows the full intake form.
    """

    lines: list[str] = []
    for index, (question, narrative_key, _prompt) in enumerate(_ISSUE_TEMPLATE, start=1):
        answer = _issue_slot_answer(index, narrative_key, narrative, task_id, environment)
        lines.append(f"{index}. {question}:")
        for line in answer.splitlines() or [answer]:
            lines.append(f"   {line}")
    return lines


def _issue_slot_answer(
    index: int,
    narrative_key: str | None,
    narrative: dict[str, str],
    task_id: str | None,
    environment: EnvironmentDiagnosticReport,
) -> str:
    """Return the answer text for slot ``index`` in :data:`_ISSUE_TEMPLATE`."""

    if index == 4:
        # Q4 combines the auto-detected tidy3d version with the user's run-mode answer.
        run_mode = narrative.get(narrative_key) if narrative_key else None
        base = f"tidy3d {environment.tidy3d_version}"
        return f"{base} - {run_mode}" if run_mode else base
    if index == 5:
        # Q5 is only fillable via `--task-id`.
        return task_id or "(not provided)"
    if narrative_key is None:
        return "(not provided)"
    return narrative.get(narrative_key) or "(not provided)"


def compose_support_report(
    *,
    environment: EnvironmentDiagnosticReport,
    connection: ConnectionDiagnosticReport | None,
    task_id: str | None = None,
    narrative: dict[str, str] | None = None,
    include_private_network_details: bool = False,
) -> SupportReport:
    """Build a :class:`SupportReport` from already-collected environment / connection reports.

    This helper is the single source of truth for populating ``privacy_mode``,
    the private-details warning, and the narrative filter. Both :func:`diagnose_report`
    and the ``tidy3d troubleshoot report`` CLI call it so the two paths cannot drift.
    """

    return SupportReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        privacy_mode="private" if include_private_network_details else "shareable",
        private_details_warning=PRIVATE_DETAILS_WARNING
        if include_private_network_details
        else None,
        task_id=task_id,
        narrative={k: v for k, v in (narrative or {}).items() if v},
        environment=environment,
        connection=connection,
    )


def diagnose_report(
    *,
    task_id: str | None = None,
    narrative: dict[str, str] | None = None,
    run_connection: bool = True,
    api_samples: int = DEFAULT_API_SAMPLES,
    timeout: float = DEFAULT_TIMEOUT,
    include_private_network_details: bool = False,
    verbose: bool = False,
) -> SupportReport:
    """Build a combined support report matching the Tidy3D issue-report template.

    Parameters
    ----------
    task_id:
        Optional task ID the user is reporting on.
    narrative:
        Free-text answers to the support template. Missing keys are omitted from the output.
    run_connection:
        If ``True`` (default), run connection diagnostics. Disable to skip network probes.
    api_samples, timeout, include_private_network_details:
        Forwarded to :func:`diagnose_connection` when ``run_connection`` is enabled.
    verbose:
        If ``True``, echo the combined report to the Tidy3D console.
    """

    environment = diagnose_environment(
        verbose=False,
        include_private_network_details=include_private_network_details,
    )
    connection = None
    if run_connection:
        connection = diagnose_connection(
            api_samples=api_samples,
            timeout=timeout,
            verbose=False,
            include_private_network_details=include_private_network_details,
        )
    report = compose_support_report(
        environment=environment,
        connection=connection,
        task_id=task_id,
        narrative=narrative,
        include_private_network_details=include_private_network_details,
    )
    if verbose:
        get_logging_console().print(report.support_text())
    return report

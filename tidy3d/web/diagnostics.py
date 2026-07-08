"""Connection diagnostics for support-facing network investigations."""

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
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field

from tidy3d.config import config
from tidy3d.log import get_logging_console
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

        lines = [
            "Tidy3D connection diagnostics",
            f"- generated_at: {self.generated_at}",
            f"- privacy_mode: {self.privacy_mode}",
            f"- tidy3d_version: {self.tidy3d_version}",
            f"- python_version: {self.python_version}",
            f"- platform: {self.platform}",
            f"- api_endpoint_host: {self.api_endpoint_host or 'unknown'}",
            f"- api_key_configured: {self.api_key_configured}",
        ]
        if self.private_details_warning:
            lines.extend(["", self.private_details_warning, ""])
        if self.configuration is not None:
            lines.extend(
                [
                    f"- ssl_verify: {self.configuration.ssl_verify}",
                    f"- ssl_version: {self.configuration.ssl_version or 'default'}",
                    f"- proxy_env: {_format_env_summary(self.configuration.proxy_environment)}",
                    f"- certificate_env: {_format_env_summary(self.configuration.certificate_environment)}",
                    f"- tidy3d_env: {_format_env_summary(self.configuration.tidy3d_environment)}",
                ]
            )
            for warning in self.configuration.warnings:
                lines.append(f"- warning: {warning}")

        lines.extend(["", "Checks:"])
        for result in self.results:
            summary = _result_summary(result)
            lines.append(f"- {result.name}: {result.status}{summary}")
            if result.recommendation:
                lines.append(f"  recommendation: {result.recommendation}")

        lines.extend(["", "JSON:", self.model_dump_json(indent=2)])
        return "\n".join(lines)


def _host_from_url(url: str | None) -> str | None:
    """Return only the host from a URL so signed URL secrets are not exposed."""

    if not url:
        return None
    parsed = urlparse(str(url))
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

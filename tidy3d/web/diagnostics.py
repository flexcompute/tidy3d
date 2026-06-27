"""Connection diagnostics for support-facing network investigations."""

from __future__ import annotations

import platform
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, Field

from tidy3d.config import config
from tidy3d.log import get_logging_console
from tidy3d.version import __version__
from tidy3d.web.core.http_util import api_key, api_key_auth

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_TIMEOUT = 20.0
DEFAULT_API_SAMPLES = 3
DEFAULT_DOWNLOAD_BYTES = 100 * 1024 * 1024
DEFAULT_DOWNLOAD_URL_ENDPOINT = "tidy3d/diagnostics/download-url"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_URL_KEYS = ("download_url", "downloadUrl", "url")


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


class ConnectionDiagnosticReport(BaseModel):
    """Support-facing network diagnostic report."""

    generated_at: str = Field(title="UTC report creation timestamp.")
    tidy3d_version: str = Field(title="Tidy3D client version.")
    python_version: str = Field(title="Python version.")
    platform: str = Field(title="Operating system and machine summary.")
    api_endpoint_host: str | None = Field(None, title="Configured API endpoint host.")
    api_key_configured: bool = Field(title="Whether an API key is configured.")
    results: tuple[ConnectionDiagnosticResult, ...] = Field(title="Diagnostic check results.")

    def support_text(self) -> str:
        """Return a paste-friendly support summary with structured JSON."""

        lines = [
            "Tidy3D connection diagnostics",
            f"- generated_at: {self.generated_at}",
            f"- tidy3d_version: {self.tidy3d_version}",
            f"- python_version: {self.python_version}",
            f"- platform: {self.platform}",
            f"- api_endpoint_host: {self.api_endpoint_host or 'unknown'}",
            f"- api_key_configured: {self.api_key_configured}",
            "",
            "Checks:",
        ]
        for result in self.results:
            summary = _result_summary(result)
            lines.append(f"- {result.name}: {result.status}{summary}")

        lines.extend(["", "JSON:", self.model_dump_json(indent=2)])
        return "\n".join(lines)


def _host_from_url(url: str | None) -> str | None:
    """Return only the host from a URL so signed URL secrets are not exposed."""

    if not url:
        return None
    return urlparse(str(url)).netloc or None


def _redact_url(url: str) -> str:
    """Strip query and fragment data from a URL."""

    parsed = urlparse(url)
    if not (parsed.scheme and parsed.netloc) and not parsed.path.startswith("/"):
        return url
    return parsed._replace(query="", fragment="").geturl()


def _sanitize_error_message(message: str) -> str:
    """Remove signed URL query strings from error text."""

    return re.sub(
        r"(?:https?://|/)[^\s)>\]\"']+", lambda match: _redact_url(match.group(0)), message
    )


def _error_result(name: str, target_url: str | None, exc: Exception) -> ConnectionDiagnosticResult:
    """Build a failed result without including signed URL query strings."""

    return ConnectionDiagnosticResult(
        name=name,
        status="fail",
        target_host=_host_from_url(target_url),
        error_type=type(exc).__name__,
        error=_sanitize_error_message(str(exc)),
    )


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


def _check_api_latency(
    *,
    session: requests.Session,
    api_url: str,
    samples: int,
    timeout: float,
) -> ConnectionDiagnosticResult:
    """Measure latency to a lightweight API endpoint."""

    try:
        measurements = []
        for _ in range(samples):

            def request_once() -> None:
                response = session.get(api_url, timeout=timeout)
                if not response.ok:
                    response.raise_for_status()
                return None

            measurements.append(_timed_sample(request_once))
        return ConnectionDiagnosticResult(
            name="api_latency",
            status="pass",
            target_host=_host_from_url(api_url),
            samples=tuple(measurements),
        )
    except Exception as exc:
        return _error_result("api_latency", api_url, exc)


def _check_authentication(
    *,
    session: requests.Session,
    api_url: str,
    timeout: float,
) -> ConnectionDiagnosticResult:
    """Check whether the configured API key can authenticate against the API."""

    try:

        def request_once() -> None:
            response = session.get(api_url, auth=api_key_auth, timeout=timeout)
            if not response.ok:
                response.raise_for_status()
            return None

        sample = _timed_sample(request_once)
        return ConnectionDiagnosticResult(
            name="authentication",
            status="pass",
            target_host=_host_from_url(api_url),
            samples=(sample,),
        )
    except Exception as exc:
        return _error_result("authentication", api_url, exc)


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
        return _error_result("storage_download", download_url, exc)


def diagnose_connection(
    *,
    api_samples: int = DEFAULT_API_SAMPLES,
    timeout: float = DEFAULT_TIMEOUT,
    verbose: bool = True,
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

    Returns
    -------
    ConnectionDiagnosticReport
        Structured diagnostic report suitable for support tickets.
    """

    if api_samples < 1:
        raise ValueError("'api_samples' must be at least 1.")
    if timeout <= 0:
        raise ValueError("'timeout' must be positive.")

    session = requests.Session()
    health_url = config.web.build_api_url("health")
    auth_url = config.web.build_api_url("tidy3d/projects")
    download_url_endpoint = config.web.build_api_url(DEFAULT_DOWNLOAD_URL_ENDPOINT)
    report = ConnectionDiagnosticReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        tidy3d_version=__version__,
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        api_endpoint_host=_host_from_url(str(config.web.api_endpoint)),
        api_key_configured=api_key() is not None,
        results=(
            _check_api_latency(
                session=session,
                api_url=health_url,
                samples=api_samples,
                timeout=timeout,
            ),
            _check_authentication(
                session=session,
                api_url=auth_url,
                timeout=timeout,
            ),
            _check_download_throughput(
                session=session,
                endpoint_url=download_url_endpoint,
                timeout=timeout,
            ),
        ),
    )
    if verbose:
        get_logging_console().print(report.support_text())
    return report

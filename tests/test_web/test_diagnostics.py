from __future__ import annotations

from tidy3d.web import diagnostics


class _FakeResponse:
    def __init__(
        self,
        chunks: tuple[bytes, ...] = (),
        json_payload=None,
        url: str | None = None,
    ) -> None:
        self.ok = True
        self.chunks = chunks
        self.closed = False
        self.json_payload = json_payload
        self.url = url

    def raise_for_status(self) -> None:
        raise AssertionError("raise_for_status should not be called for successful responses")

    def json(self):
        return self.json_payload

    def iter_content(self, chunk_size: int):
        yield from self.chunks

    def close(self) -> None:
        self.closed = True


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
        else:
            response = _FakeResponse(
                chunks=self.download_chunks if kwargs.get("stream") else (),
                url=url,
            )
        self.responses.append(response)
        return response


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


def test_diagnose_connection_uses_default_storage_download_url(monkeypatch):
    """Diagnostic report should always resolve and use the default storage test URL."""
    session = _FakeSession()
    monkeypatch.setattr(diagnostics.requests, "Session", lambda: session)
    monkeypatch.setattr(diagnostics, "api_key", lambda: "configured")

    report = diagnostics.diagnose_connection(
        api_samples=2,
        verbose=False,
    )

    assert report.api_key_configured is True
    assert [result.name for result in report.results] == [
        "api_latency",
        "authentication",
        "storage_download",
    ]
    assert report.results[0].status == "pass"
    assert len(report.results[0].samples) == 2
    assert report.results[1].status == "pass"
    assert report.results[2].status == "pass"
    assert report.results[2].target_host == "storage.example.com"
    assert session.calls[-2][0].endswith(diagnostics.DEFAULT_DOWNLOAD_URL_ENDPOINT)
    assert session.calls[-1][0] == _FakeSession.download_url
    assert len(session.calls) == 5


def test_diagnose_connection_measures_storage_download_and_redacts_url(monkeypatch):
    """Storage throughput should read bytes while keeping signed URL secrets out of reports."""
    session = _FakeSession()
    monkeypatch.setattr(diagnostics.requests, "Session", lambda: session)
    monkeypatch.setattr(diagnostics, "api_key", lambda: "configured")

    report = diagnostics.diagnose_connection(
        api_samples=1,
        verbose=False,
    )

    storage_result = report.results[2]
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

    report = diagnostics.diagnose_connection(
        api_samples=1,
        verbose=False,
    )

    storage_result = report.results[2]
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

    report = diagnostics.diagnose_connection(
        api_samples=1,
        verbose=False,
    )

    storage_result = report.results[2]
    assert storage_result.status == "fail"
    assert storage_result.error == "failed to download https://storage.example.com/diagnostic.bin"
    assert "secret" not in report.model_dump_json()


def test_diagnose_connection_redacts_relative_signed_url_from_errors(monkeypatch):
    """urllib3-style relative URL errors should not expose signed URL query strings."""
    session = _RelativeUrlFailingDownloadSession()
    monkeypatch.setattr(diagnostics.requests, "Session", lambda: session)
    monkeypatch.setattr(diagnostics, "api_key", lambda: "configured")

    report = diagnostics.diagnose_connection(
        api_samples=1,
        verbose=False,
    )

    storage_result = report.results[2]
    assert storage_result.status == "fail"
    assert storage_result.error == "Max retries exceeded with url: /diagnostic.bin"
    assert "secret" not in report.model_dump_json()
    assert "X-Amz-Signature" not in report.support_text()

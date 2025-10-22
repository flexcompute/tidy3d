from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import tidy3d
from tidy3d.web.core import s3utils


@pytest.fixture
def mock_S3STSToken(monkeypatch):
    mock_token = MagicMock()
    mock_token.cloud_path = ""
    mock_token.user_credential = ""
    mock_token.get_bucket = lambda: ""
    mock_token.get_s3_key = lambda: ""
    mock_token.is_expired = lambda: False
    mock_token.get_client = lambda: tidy3d.web.core.s3utils.boto3.client()
    monkeypatch.setattr(
        target=tidy3d.web.core.s3utils, name="_S3STSToken", value=MagicMock(return_value=mock_token)
    )
    return mock_token


@pytest.fixture
def mock_get_s3_sts_token(monkeypatch):
    def _mock_get_s3_sts_token(resource_id, remote_filename):
        return s3utils._S3STSToken(resource_id, remote_filename)

    monkeypatch.setattr(
        target=tidy3d.web.core.s3utils, name="get_s3_sts_token", value=_mock_get_s3_sts_token
    )
    return _mock_get_s3_sts_token


@pytest.fixture
def mock_s3_client(monkeypatch):
    """
    Fixture that provides a generic mock S3 client.
    Method-specific side_effects are omitted here and are specified later in the unit tests.
    """
    mock_client = MagicMock()
    # Patch the `client` as it is imported within `tidy3d.web.core.s3utils.boto3` so that
    # whenever it's invoked (for example with "s3"), it returns our `mock_client`.
    monkeypatch.setattr(
        target=tidy3d.web.core.s3utils.boto3,
        name="client",
        value=MagicMock(return_value=mock_client),
    )
    return mock_client


def test_download_s3_file_success(mock_s3_client, mock_get_s3_sts_token, mock_S3STSToken, tmp_path):
    """Tests a successful download."""
    destination_path = tmp_path / "downloaded_file.txt"
    expected_content = "abcdefg"

    def simulate_download_success(Bucket, Key, Filename, Callback, Config, **kwargs):
        with open(Filename, "w") as f:
            f.write(expected_content)
        return None

    mock_s3_client.download_file.side_effect = simulate_download_success
    mock_S3STSToken.get_bucket = lambda: "test-bucket"
    mock_S3STSToken.get_s3_key = lambda: "test-key"

    s3utils.download_file(
        resource_id="1234567890",
        remote_filename=destination_path.name,
        to_file=str(destination_path),
        verbose=False,
        progress_callback=None,
    )

    # Check that mock_s3_client.download_file() was invoked with the correct arguments.
    mock_s3_client.download_file.assert_called_once()
    _call_args, call_kwargs = mock_s3_client.download_file.call_args
    assert call_kwargs["Bucket"] == "test-bucket"
    assert call_kwargs["Key"] == "test-key"
    assert call_kwargs["Filename"].endswith(s3utils.IN_TRANSIT_SUFFIX)
    assert destination_path.exists()
    with open(destination_path) as f:
        assert f.read() == expected_content
    for p in destination_path.parent.iterdir():
        assert not p.name.endswith(s3utils.IN_TRANSIT_SUFFIX)  # no temporary files are present


def test_download_s3_file_raises_oserror(
    mock_s3_client, mock_get_s3_sts_token, mock_S3STSToken, tmp_path
):
    """Tests download failing with an ``OSError`` (No space left on device)."""
    destination_path = tmp_path / "downloaded_file.txt"

    def simulate_download_failure(Bucket, Key, Filename, Callback, Config, **kwargs):
        with open(Filename, "w") as f:
            f.write("abc")
        raise OSError("No space left on device")

    mock_s3_client.download_file.side_effect = simulate_download_failure
    mock_S3STSToken.get_bucket = lambda: "test-bucket"
    mock_S3STSToken.get_s3_key = lambda: "test-key"

    with pytest.raises(OSError, match="No space left on device"):
        s3utils.download_file(
            resource_id="1234567890",
            remote_filename=destination_path.name,
            to_file=str(destination_path),
            verbose=False,
            progress_callback=None,
        )

    # Check that mock_s3_client.download_file() was invoked with the correct arguments.
    mock_s3_client.download_file.assert_called_once()
    _call_args, call_kwargs = mock_s3_client.download_file.call_args
    assert call_kwargs["Bucket"] == "test-bucket"
    assert call_kwargs["Key"] == "test-key"
    assert call_kwargs["Filename"].endswith(s3utils.IN_TRANSIT_SUFFIX)
    # Since downloading failed, no new files should exist locally.
    assert not destination_path.exists()
    for p in destination_path.parent.iterdir():
        assert not p.name.endswith(s3utils.IN_TRANSIT_SUFFIX)  # no temporary files are present

"""handles filesystem, storage"""

from __future__ import annotations

import os
import tempfile
import urllib
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Optional

import boto3
import rich
from boto3.s3.transfer import TransferConfig
from pydantic import BaseModel, Field
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from tidy3d.config import config

from .core_config import get_logger_console
from .exceptions import WebError
from .file_util import extract_gzip_file
from .http_util import http

IN_TRANSIT_SUFFIX = ".tmp"


class _UserCredential(BaseModel):
    """Stores information about user credentials."""

    access_key_id: str = Field(alias="accessKeyId")
    expiration: datetime
    secret_access_key: str = Field(alias="secretAccessKey")
    session_token: str = Field(alias="sessionToken")


class _S3STSToken(BaseModel):
    """Stores information about S3 token."""

    cloud_path: str = Field(alias="cloudpath")
    user_credential: _UserCredential = Field(alias="userCredentials")

    def get_bucket(self) -> str:
        """Get the bucket name for this token."""

        r = urllib.parse.urlparse(self.cloud_path)
        return r.netloc

    def get_s3_key(self) -> str:
        """Get the s3 key for this token."""

        r = urllib.parse.urlparse(self.cloud_path)
        return r.path[1:]

    def get_client(self) -> boto3.client:
        """Get the boto client for this token.

        Automatically configures custom S3 endpoint if specified in web.env_vars.
        """

        client_kwargs = {
            "service_name": "s3",
            "region_name": config.web.s3_region,
            "aws_access_key_id": self.user_credential.access_key_id,
            "aws_secret_access_key": self.user_credential.secret_access_key,
            "aws_session_token": self.user_credential.session_token,
            "verify": config.web.ssl_verify,
        }

        # Add custom S3 endpoint if configured (e.g., for Nexus deployments)
        if config.web.env_vars and "AWS_ENDPOINT_URL_S3" in config.web.env_vars:
            s3_endpoint = config.web.env_vars["AWS_ENDPOINT_URL_S3"]
            client_kwargs["endpoint_url"] = s3_endpoint

        return boto3.client(**client_kwargs)

    def is_expired(self) -> bool:
        """True if token is expired."""

        return (
            self.user_credential.expiration
            - datetime.now(tz=self.user_credential.expiration.tzinfo)
        ).total_seconds() < 300


class UploadProgress:
    """Updates progressbar with the upload status.

    Attributes
    ----------
    progress : rich.progress.Progress()
        Progressbar instance from rich
    ul_task : rich.progress.Task
        Progressbar task instance.
    """

    def __init__(self, size_bytes: int, progress: rich.progress.Progress) -> None:
        """initialize with the size of file and rich.progress.Progress() instance.

        Parameters
        ----------
        size_bytes: int
            Number of total bytes to upload.
        progress : rich.progress.Progress()
            Progressbar instance from rich
        """
        self.progress = progress
        self.ul_task = self.progress.add_task("[red]Uploading...", total=size_bytes)

    def report(self, bytes_in_chunk: Any) -> None:
        """Update the progressbar with the most recent chunk.

        Parameters
        ----------
        bytes_in_chunk : int
            Description
        """
        self.progress.update(self.ul_task, advance=bytes_in_chunk)


class DownloadProgress:
    """Updates progressbar using the download status.

    Attributes
    ----------
    progress : rich.progress.Progress()
        Progressbar instance from rich
    ul_task : rich.progress.Task
        Progressbar task instance.
    """

    def __init__(self, size_bytes: int, progress: rich.progress.Progress) -> None:
        """initialize with the size of file and rich.progress.Progress() instance

        Parameters
        ----------
        size_bytes: float
            Number of total bytes to download.
        progress : rich.progress.Progress()
            Progressbar instance from rich
        """
        self.progress = progress
        self.dl_task = self.progress.add_task("[red]Downloading...", total=size_bytes)

    def report(self, bytes_in_chunk: int) -> None:
        """Update the progressbar with the most recent chunk.

        Parameters
        ----------
        bytes_in_chunk : float
            Description
        """
        self.progress.update(self.dl_task, advance=bytes_in_chunk)


class _S3Action(Enum):
    UPLOADING = "↑"
    DOWNLOADING = "↓"


def _get_progress(action: _S3Action) -> Progress:
    """Get the progress of an action."""

    col = (
        TextColumn(f"[bold green]{_S3Action.DOWNLOADING.value}")
        if action == _S3Action.DOWNLOADING
        else TextColumn(f"[bold red]{_S3Action.UPLOADING.value}")
    )
    return Progress(
        col,
        TextColumn("[bold blue]{task.fields[filename]}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
        console=get_logger_console(),
    )


_s3_config = TransferConfig()

_s3_sts_tokens: dict[str, _S3STSToken] = {}


def get_s3_sts_token(
    resource_id: str, file_name: PathLike, extra_arguments: Optional[Mapping[str, str]] = None
) -> _S3STSToken:
    """Get s3 sts token for the given resource id and file name.

    Parameters
    ----------
    resource_id : str
        The resource id, e.g. task id.
    file_name : PathLike
        The remote file name on S3.
    extra_arguments : Mapping[str, str]
        Additional arguments for the query url.

    Returns
    -------
    _S3STSToken
        The S3 STS token.
    """
    file_name = str(Path(file_name).as_posix())
    cache_key = f"{resource_id}:{file_name}"
    if cache_key not in _s3_sts_tokens or _s3_sts_tokens[cache_key].is_expired():
        method = f"tidy3d/py/tasks/{resource_id}/file?filename={file_name}"
        if extra_arguments is not None:
            method += "&" + "&".join(f"{k}={v}" for k, v in extra_arguments.items())
        resp = http.get(method)
        token = _S3STSToken.model_validate(resp)
        _s3_sts_tokens[cache_key] = token
    return _s3_sts_tokens[cache_key]


def upload_file(
    resource_id: str,
    path: PathLike,
    remote_filename: PathLike,
    verbose: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None,
    extra_arguments: Optional[Mapping[str, str]] = None,
) -> None:
    """Upload a file to S3.

    Parameters
    ----------
    resource_id : str
        The resource id, e.g. task id.
    path : PathLike
        Path to the file to upload.
    remote_filename : PathLike
        The remote file name on S3 relative to the resource context root path.
    verbose : bool = True
        Whether to display a progressbar for the upload.
    progress_callback : Callable[[float], None] = None
        User-supplied callback function with ``bytes_in_chunk`` as argument.
    extra_arguments : Mapping[str, str]
        Additional arguments used to specify the upload bucket.
    """

    path = Path(path)
    token = get_s3_sts_token(resource_id, remote_filename, extra_arguments)

    def _upload(_callback: Callable) -> None:
        """Perform the upload with a callback function.

        Parameters
        ----------
        _callback : Callable[[float], None]
            Callback function for upload, accepts ``bytes_in_chunk``
        """

        with path.open("rb") as data:
            token.get_client().upload_fileobj(
                data,
                Bucket=token.get_bucket(),
                Key=token.get_s3_key(),
                Callback=_callback,
                Config=_s3_config,
                ExtraArgs={"ContentEncoding": "gzip"}
                if token.get_s3_key().endswith(".gz")
                else None,
            )

    if progress_callback is not None:
        _upload(progress_callback)
    else:
        if verbose:
            with _get_progress(_S3Action.UPLOADING) as progress:
                total_size = path.stat().st_size
                task_id = progress.add_task(
                    "upload", filename=str(remote_filename), total=total_size
                )

                def _callback(bytes_in_chunk: int) -> None:
                    progress.update(task_id, advance=bytes_in_chunk)

                _upload(_callback)

                progress.update(task_id, completed=total_size, refresh=True)

        else:
            _upload(lambda bytes_in_chunk: None)


def download_file(
    resource_id: str,
    remote_filename: PathLike,
    to_file: Optional[PathLike] = None,
    verbose: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Path:
    """Download file from S3.

    Parameters
    ----------
    resource_id : str
        The resource id, e.g. task id.
    remote_filename : PathLike
        Path to the remote file.
    to_file : PathLike = None
        Local filename to save to; if not specified, defaults to ``remote_filename`` in a
        directory named after ``resource_id``.
    verbose : bool = True
        Whether to display a progressbar for the upload
    progress_callback : Callable[[float], None] = None
        User-supplied callback function with ``bytes_in_chunk`` as argument.
    """

    token = get_s3_sts_token(resource_id, remote_filename)
    client = token.get_client()
    meta_data = client.head_object(Bucket=token.get_bucket(), Key=token.get_s3_key())

    # Get only last part of the remote file name
    remote_basename = Path(remote_filename).name

    # set to_file if None
    if to_file is None:
        to_path = Path(resource_id) / remote_basename
    else:
        to_path = Path(to_file)

    # make the leading directories in the 'to_path', if any
    to_path.parent.mkdir(parents=True, exist_ok=True)

    def _download(_callback: Callable) -> None:
        """Perform the download with a callback function.

        Parameters
        ----------
        _callback : Callable[[float], None]
            Callback function for download, accepts ``bytes_in_chunk``
        """
        # Caller can assume the existence of the file means download succeeded.
        # So make sure this file does not exist until that assumption is true.
        to_path.unlink(missing_ok=True)
        # Download to a temporary file.
        try:
            fd, tmp_file_path_str = tempfile.mkstemp(suffix=IN_TRANSIT_SUFFIX, dir=to_path.parent)
            os.close(fd)  # `tempfile.mkstemp()` creates and opens a randomly named file.  close it.
            to_path_tmp = Path(tmp_file_path_str)
            client.download_file(
                Bucket=token.get_bucket(),
                Filename=str(to_path_tmp),
                Key=token.get_s3_key(),
                Callback=_callback,
                Config=_s3_config,
            )
            to_path_tmp.rename(to_path)
        except Exception as e:
            to_path_tmp.unlink(missing_ok=True)  # Delete incompletely downloaded file.
            raise e

    if progress_callback is not None:
        _download(progress_callback)
    else:
        if verbose:
            with _get_progress(_S3Action.DOWNLOADING) as progress:
                total_size = meta_data.get("ContentLength", 0)
                progress.start()
                task_id = progress.add_task("download", filename=remote_basename, total=total_size)

                def _callback(bytes_in_chunk: int) -> None:
                    progress.update(task_id, advance=bytes_in_chunk)

                _download(_callback)

                progress.update(task_id, completed=total_size, refresh=True)

        else:
            _download(lambda bytes_in_chunk: None)

    return to_path


def download_gz_file(
    resource_id: str,
    remote_filename: PathLike,
    to_file: Optional[PathLike] = None,
    verbose: bool = True,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Path:
    """Download a ``.gz`` file and unzip it into ``to_file``, unless ``to_file`` itself
    ends in .gz

    Parameters
    ----------
    resource_id : str
        The resource id, e.g. task id.
    remote_filename : PathLike
        Path to the remote file.
    to_file : Optional[PathLike] = None
        Local filename to save to; if not specified, defaults to ``remote_filename`` with the
        ``.gz`` suffix removed in a directory named after ``resource_id``.
    verbose : bool = True
        Whether to display a progressbar for the upload
    progress_callback : Callable[[float], None] = None
        User-supplied callback function with ``bytes_in_chunk`` as argument.
    """

    # If to_file is a gzip extension, just download
    if to_file is None:
        remote_basename = Path(remote_filename).name
        if remote_basename.endswith(".gz"):
            remote_basename = remote_basename[:-3]
        to_path = Path(resource_id) / remote_basename
    else:
        to_path = Path(to_file)

    suffixes = "".join(to_path.suffixes).lower()
    if suffixes.endswith(".gz"):
        return download_file(
            resource_id,
            remote_filename,
            to_file=to_path,
            verbose=verbose,
            progress_callback=progress_callback,
        )

    # Otherwise, download and unzip
    # The tempfile is set as ``hdf5.gz`` so that the mock download in the webapi tests works
    tmp_file, tmp_file_path_str = tempfile.mkstemp(".hdf5.gz")
    os.close(tmp_file)

    # make the leading directories in the 'to_file', if any
    to_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        download_file(
            resource_id,
            remote_filename,
            to_file=Path(tmp_file_path_str),
            verbose=verbose,
            progress_callback=progress_callback,
        )
        if not Path(tmp_file_path_str).exists():
            raise WebError(f"Failed to download and extract '{remote_filename}'.")

        tmp_out_fd, tmp_out_path_str = tempfile.mkstemp(
            suffix=IN_TRANSIT_SUFFIX, dir=to_path.parent
        )
        os.close(tmp_out_fd)
        tmp_out_path = Path(tmp_out_path_str)
        try:
            extract_gzip_file(Path(tmp_file_path_str), tmp_out_path)
            tmp_out_path.replace(to_path)
        except Exception as e:
            tmp_out_path.unlink(missing_ok=True)
            raise WebError(
                f"Failed to extract '{remote_filename}' from '{tmp_file_path_str}' to '{to_path}'."
            ) from e
    finally:
        Path(tmp_file_path_str).unlink(missing_ok=True)
    return to_path

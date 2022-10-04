""" handles filesystem, storage """
import io
import os
import urllib
from datetime import datetime
from enum import Enum

import boto3
from boto3.s3.transfer import TransferConfig
from pydantic import BaseModel, Field
from rich.progress import TextColumn, Progress, BarColumn, DownloadColumn
from rich.progress import TransferSpeedColumn, TimeRemainingColumn

from . import httputils as http
from .config import DEFAULT_CONFIG


class _UserCredential(BaseModel):
    access_key_id: str = Field(alias="accessKeyId")
    expiration: datetime
    secret_access_key: str = Field(alias="secretAccessKey")
    session_token: str = Field(alias="sessionToken")


class _S3STSToken(BaseModel):
    cloud_path: str = Field(alias="cloudpath")
    user_credential: _UserCredential = Field(alias="userCredentials")

    def get_bucket(self) -> str:
        """
        @return: bucket name
        """
        r = urllib.parse.urlparse(self.cloud_path)
        return r.netloc

    def get_s3_key(self) -> str:
        """@return: s3 key"""
        r = urllib.parse.urlparse(self.cloud_path)
        return r.path[1:]

    def get_client(self) -> boto3.client:
        """
        @return: boto3 client

        """
        return boto3.client(
            "s3",
            region_name=DEFAULT_CONFIG.s3_region,
            aws_access_key_id=self.user_credential.access_key_id,
            aws_secret_access_key=self.user_credential.secret_access_key,
            aws_session_token=self.user_credential.session_token,
        )

    def is_expired(self) -> bool:
        """
        @return: True if token is expired
        @return:
        """
        return (
            self.user_credential.expiration
            - datetime.now(tz=self.user_credential.expiration.tzinfo)
        ).total_seconds() < 300


class UploadProgress:
    """updates progressbar for the upload status"""

    def __init__(self, size_bytes, progress):
        """initialize with the size of file and rich.progress.Progress() instance"""
        self.progress = progress
        self.ul_task = self.progress.add_task("[red]Uploading...", total=size_bytes)

    def report(self, bytes_in_chunk):
        """the progressbar with recent chunk"""
        self.progress.update(self.ul_task, advance=bytes_in_chunk)


class DownloadProgress:
    """updates progressbar for the download status"""

    def __init__(self, size_bytes, progress):
        """initialize with the size of file and rich.progress.Progress() instance"""
        self.progress = progress
        self.dl_task = self.progress.add_task("[red]Downloading...", total=size_bytes)

    def report(self, bytes_in_chunk):
        """the progressbar with recent chunk"""
        self.progress.update(self.dl_task, advance=bytes_in_chunk)


class _S3Action(Enum):
    UPLOADING = "↑"
    DOWNLOADING = "↓"


def _get_progress(action: _S3Action):
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
    )


_s3_config = TransferConfig(
    multipart_threshold=1024 * 25,
    max_concurrency=50,
    multipart_chunksize=1024 * 25,
    use_threads=True,
)

_s3_sts_tokens: [str, _S3STSToken] = {}


def get_s3_sts_token(resource_id: str, file_name: str) -> _S3STSToken:
    """
    get s3 sts token for the given resource id and file name
    @param resource_id: the resource id, e.g. task id"
    @param file_name: the remote file name on S3
    @return: _S3STSToken
    """
    cache_key = f"{resource_id}:{file_name}"
    if cache_key not in _s3_sts_tokens or _s3_sts_tokens[cache_key].is_expired():
        method = f"tidy3d/tasks/{resource_id}/file?filename={file_name}"
        resp = http.get(method)
        token = _S3STSToken.parse_obj(resp)
        _s3_sts_tokens[cache_key] = token
    return _s3_sts_tokens[cache_key]


def upload_string(resource_id: str, content: str, remote_filename: str):
    """
    upload a string to a file on S3
    @param resource_id: the resource id, e.g. task id
    @param content:     the content of the file
    @param remote_filename: the remote file name on S3
    """
    with _get_progress(_S3Action.UPLOADING) as progress:
        task_id = progress.add_task("upload", filename=remote_filename, total=len(content))

        def _call_back(bytes_in_chunk):
            progress.update(task_id, advance=bytes_in_chunk)

        token = get_s3_sts_token(resource_id, remote_filename)
        token.get_client().upload_fileobj(
            io.BytesIO(content.encode("utf-8")),
            Bucket=token.get_bucket(),
            Key=token.get_s3_key(),
            Callback=_call_back,
            Config=_s3_config,
        )


def upload_file(resource_id: str, path: str, remote_filename: str):
    """
    upload file to S3
    @param resource_id: the resource id, e.g. task id
    @param path: path to the file
    @param remote_filename: the remote file name on S3
    """
    with _get_progress(_S3Action.UPLOADING) as progress:
        task_id = progress.add_task("upload", filename=remote_filename, total=os.path.getsize(path))

        def _call_back(bytes_in_chunk):
            progress.update(task_id, advance=bytes_in_chunk)

        token = get_s3_sts_token(resource_id, remote_filename)
        with open(path, "rb") as data:
            token.get_client().upload_fileobj(
                data,
                Bucket=token.get_bucket(),
                Key=token.get_s3_key(),
                Callback=_call_back,
                Config=_s3_config,
            )


def download_file(resource_id: str, remote_filename: str, to_file: str = None):
    """
    download file from S3
    @param resource_id: the resource id, e.g. task id
    @param remote_filename: the remote file name on S3
    @param to_file: the local file name to save the file
    """
    token = get_s3_sts_token(resource_id, remote_filename)
    client = token.get_client()
    meta_data = client.head_object(Bucket=token.get_bucket(), Key=token.get_s3_key())
    with _get_progress(_S3Action.DOWNLOADING) as progress:
        progress.start()
        task_id = progress.add_task(
            "download",
            filename=os.path.basename(remote_filename),
            total=meta_data.get("ContentLength", 0),
        )

        def _call_back(bytes_in_chunk):
            progress.update(task_id, advance=bytes_in_chunk)

        if not to_file:
            os.makedirs(resource_id, exist_ok=True)
            to_file = os.path.join(resource_id, os.path.basename(remote_filename))

        client.download_file(
            Bucket=token.get_bucket(), Filename=to_file, Key=token.get_s3_key(), Callback=_call_back
        )

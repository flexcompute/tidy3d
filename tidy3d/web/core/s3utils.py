"""Compatibility shim for :mod:`tidy3d._common.web.core.s3utils`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.web.core.s3utils import (
    IN_TRANSIT_SUFFIX,
    DownloadProgress,
    UploadProgress,
    _get_progress,
    _s3_config,
    _s3_sts_tokens,
    _S3Action,
    _S3STSToken,
    _UserCredential,
    download_file,
    download_gz_file,
    get_s3_sts_token,
    upload_file,
)

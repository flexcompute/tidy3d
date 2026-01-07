"""Compatibility shim for :mod:`tidy3d._common.web.core.http_util`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.web.core.http_util import (
    HttpSessionManager,
    JSONType,
    ResponseCodes,
    TLSAdapter,
    api_key,
    api_key_auth,
    get_headers,
    get_user_agent,
    get_version,
    http,
    http_interceptor,
)

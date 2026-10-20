"""Local caches."""

from __future__ import annotations

FOLDER_CACHE = {}
_folder_cache_scope = None


def scope_folder_cache() -> None:
    """Do not reuse folder IDs after changing service or account credentials."""
    import hashlib

    from tidy3d.config import config
    from tidy3d.web.core.http_util import api_key

    global _folder_cache_scope
    scope = (str(config.web.api_endpoint), hashlib.sha256((api_key() or "").encode()).digest())
    if scope != _folder_cache_scope:
        FOLDER_CACHE.clear()
        _folder_cache_scope = scope


S3_STS_TOKENS = {}

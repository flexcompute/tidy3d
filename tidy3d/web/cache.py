"""Compatibility shim for :mod:`tidy3d._common.web.cache`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.web.cache import (
    _CACHE,
    CACHE_ARTIFACT_NAME,
    CACHE_METADATA_NAME,
    CACHE_STATS_NAME,
    TMP_BATCH_PREFIX,
    TMP_PREFIX,
    CacheEntry,
    CacheEntryMetadata,
    CacheStats,
    LocalCache,
    _canonicalize,
    _copy_and_hash,
    _Hasher,
    _now,
    _read_metadata,
    _store_mode_solver_in_cache,
    _timestamp_suffix,
    _write_metadata,
    build_cache_key,
    build_entry_metadata,
    clear,
    get_cache_entry_dir,
    resolve_local_cache,
)

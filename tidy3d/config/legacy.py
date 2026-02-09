"""Compatibility shim for :mod:`tidy3d._common.config.legacy`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.config.legacy import (
    LegacyConfigWrapper,
    LegacyEnvironment,
    LegacyEnvironmentConfig,
    _maybe_str,
    _warn_env_deprecated,
    finalize_legacy_migration,
    load_legacy_flat_config,
)

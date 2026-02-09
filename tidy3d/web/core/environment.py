"""Compatibility shim for :mod:`tidy3d._common.web.core.environment`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.web.core.environment import (
    _DEPRECATION_MESSAGE,
    _LEGACY_ENV_NAMES,
    Env,
    Environment,
    EnvironmentConfig,
    _get_legacy_env,
    dev,
    nexus,
    pre,
    prod,
    uat,
)

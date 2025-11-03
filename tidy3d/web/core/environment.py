"""Legacy re-export of configuration environment helpers."""

from __future__ import annotations

# TODO(FXC-3827): Remove this module-level legacy shim in Tidy3D 2.12.
import warnings
from typing import Any

from tidy3d.config import Env, Environment, EnvironmentConfig

__all__ = [  # noqa: F822
    "Env",
    "Environment",
    "EnvironmentConfig",
    "dev",
    "nexus",
    "pre",
    "prod",
    "uat",
]

_LEGACY_ENV_NAMES = {"dev", "uat", "pre", "prod", "nexus"}
_DEPRECATION_MESSAGE = (
    "'tidy3d.web.core.environment.{name}' is deprecated and will be removed in "
    "Tidy3D 2.12. Transition to 'tidy3d.config.Env.{name}' or "
    "'tidy3d.config.config.switch_profile(...)'."
)


def _get_legacy_env(name: str) -> Any:
    warnings.warn(_DEPRECATION_MESSAGE.format(name=name), DeprecationWarning, stacklevel=2)
    return getattr(Env, name)


def __getattr__(name: str) -> Any:
    if name in _LEGACY_ENV_NAMES:
        return _get_legacy_env(name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(set(__all__))

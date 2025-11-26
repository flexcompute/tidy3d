"""Base class for configuring RF and microwave models."""

from __future__ import annotations

from typing import Any

from pydantic import model_validator

from tidy3d.compat import Self
from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.config import config
from tidy3d.log import log


class MicrowaveBaseModel(Tidy3dBaseModel):
    """Base model that all RF and microwave specific components inherit from."""

    @model_validator(mode="before")
    @classmethod
    def _warn_rf_license(cls, values: dict[str, Any]) -> dict[str, Any]:
        from tidy3d.config import config

        # Skip warning when globally suppressed via config
        if not config.microwave.suppress_rf_license_warning:
            log.warning(
                "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. "
                "You have instantiated at least one RF-specific component.",
                log_once=True,
            )
        return values

    @classmethod
    def _default_without_license_warning(cls) -> Self:
        """Internal helper factory function for classes inheriting from ``MicrowaveBaseModel``."""
        if config.microwave.suppress_rf_license_warning is True:
            return cls()
        else:
            config.microwave.suppress_rf_license_warning = True
            default_constructed = cls()
            config.microwave.suppress_rf_license_warning = False
            return default_constructed

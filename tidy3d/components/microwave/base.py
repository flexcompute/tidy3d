"""Base class for configuring RF and microwave models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tidy3d.components.base import Tidy3dBaseModel
from tidy3d.config import config

if TYPE_CHECKING:
    from tidy3d.compat import Self


class MicrowaveBaseModel(Tidy3dBaseModel):
    """Base model that all RF and microwave specific components inherit from."""

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

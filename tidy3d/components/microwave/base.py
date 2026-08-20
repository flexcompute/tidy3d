"""Base class for configuring RF and microwave models."""

from __future__ import annotations

from tidy3d.components.base import Tidy3dBaseModel


class MicrowaveBaseModel(Tidy3dBaseModel):
    """Base model that all RF and microwave specific components inherit from."""

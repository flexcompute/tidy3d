"""Defines heat material specifications"""

from __future__ import annotations

from abc import ABC

from tidy3d.components.base import Tidy3dBaseModel


class HeatChargeBC(ABC, Tidy3dBaseModel):
    """Abstract heat-charge boundary conditions."""

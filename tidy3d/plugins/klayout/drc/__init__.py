"""
This module provides integration between Tidy3D and KLayout's Design Rule Check (DRC) engine, allowing you to perform design rule checks on GDS files and Tidy3D objects.

For more information, please see the README.md file in this directory.
For a quickstart example, please see `this notebook <link/to/notebook>`_.
"""

from __future__ import annotations

from .drc import (
    DRCConfig,
    DRCRunner,
    run_drc_on_gds,
)
from .results import DRCResults

__all__ = [
    "DRCConfig",
    "DRCResults",
    "DRCRunner",
    "run_drc_on_gds",
]

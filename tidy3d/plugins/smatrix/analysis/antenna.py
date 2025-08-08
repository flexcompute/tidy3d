"""Deprecation shim for antenna analysis; moved to `tidy3d.plugins.rf`."""

from __future__ import annotations

import warnings

from tidy3d.plugins.rf.analysis.antenna import (
    get_antenna_metrics_data as _get_antenna_metrics_data,
)

__all__ = ["get_antenna_metrics_data"]

warnings.warn(
    "tidy3d.plugins.smatrix.analysis.antenna.get_antenna_metrics_data is deprecated; "
    "use tidy3d.plugins.rf.analysis.antenna.get_antenna_metrics_data",
    DeprecationWarning,
    stacklevel=2,
)

get_antenna_metrics_data = _get_antenna_metrics_data

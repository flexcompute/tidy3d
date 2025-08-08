from __future__ import annotations

import pydantic.v1 as pd

from tidy3d.components.data.data_array import DataArray
from tidy3d.log import log


class PortDataArray(DataArray):
    __slots__ = ()
    _dims = ("f", "port")

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values


class TerminalPortDataArray(DataArray):
    __slots__ = ()
    _dims = ("f", "port_out", "port_in")
    _data_attrs = {"long_name": "terminal-based port matrix element"}

    @pd.root_validator(pre=False)
    def _warn_rf_license(cls, values):
        log.warning(
            "ℹ️ ⚠️ RF simulations are subject to new license requirements in the future. You have instantiated at least one RF-specific component.",
            log_once=True,
        )
        return values

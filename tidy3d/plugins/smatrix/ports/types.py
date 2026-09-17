from __future__ import annotations

from tidy3d._rf_migration import missing_rf_attribute
from tidy3d.plugins.smatrix.ports.modal import Port

PortType = Port

_MIGRATED_RF_NAMES = {
    "LumpedPortType",
    "PortCurrentType",
    "PortVoltageType",
    "TerminalPortType",
    "WavePortType",
}


def __getattr__(name: str) -> None:
    if name in _MIGRATED_RF_NAMES:
        missing_rf_attribute(__name__, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

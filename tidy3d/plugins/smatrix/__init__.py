"""Imports for the modal scattering-matrix plugin."""

from __future__ import annotations

from tidy3d.plugins.smatrix.component_modelers.base import AbstractComponentModeler
from tidy3d.plugins.smatrix.component_modelers.modal import ModalComponentModeler
from tidy3d.plugins.smatrix.component_modelers.types import ComponentModelerType
from tidy3d.plugins.smatrix.data.data_array import ModalPortDataArray
from tidy3d.plugins.smatrix.data.modal import ModalComponentModelerData
from tidy3d.plugins.smatrix.data.types import ComponentModelerDataType
from tidy3d.plugins.smatrix.ports.modal import AstigmaticGaussianPort, GaussianPort, Port

ComponentModeler = ModalComponentModeler

_MIGRATED_RF_NAMES = {
    "CoaxialLumpedPort",
    "DirectivityMonitorSpec",
    "ImpedanceSpec",
    "LumpedPort",
    "MicrowaveSMatrixData",
    "ModelerLowFrequencySmoothingSpec",
    "PortDataArray",
    "TerminalComponentModeler",
    "TerminalComponentModelerData",
    "TerminalPortDataArray",
    "TerminalWavePort",
    "WavePort",
}


def __getattr__(name: str) -> None:
    if name in _MIGRATED_RF_NAMES:
        from tidy3d._rf_migration import missing_rf_attribute

        missing_rf_attribute(__name__, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AbstractComponentModeler",
    "AstigmaticGaussianPort",
    "ComponentModeler",
    "ComponentModelerDataType",
    "ComponentModelerType",
    "GaussianPort",
    "ModalComponentModeler",
    "ModalComponentModelerData",
    "ModalPortDataArray",
    "Port",
]

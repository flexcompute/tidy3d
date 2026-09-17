from __future__ import annotations

from pydantic import NonNegativeInt

from tidy3d._rf_migration import missing_rf_attribute

# S matrix indices and entries for the ModalComponentModeler
MatrixIndex = tuple[str, NonNegativeInt]  # the 'i' in S_ij
Element = tuple[MatrixIndex, MatrixIndex]  # the 'ij' in S_ij

_MIGRATED_RF_NAMES = {"NetworkElement", "NetworkIndex", "SParamDef"}


def __getattr__(name: str) -> None:
    if name in _MIGRATED_RF_NAMES:
        missing_rf_attribute(__name__, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

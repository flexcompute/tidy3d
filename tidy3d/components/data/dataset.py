"""Compatibility shim for :mod:`tidy3d._common.components.data.dataset`."""

# ruff: noqa: F401 - ignore unused imports, imports ensure compatibility

# marked as migrated to _common
from __future__ import annotations

from tidy3d._common.components.data.dataset import (
    DEFAULT_MAX_CELLS_PER_STEP,
    DEFAULT_MAX_SAMPLES_PER_STEP,
    DEFAULT_TOLERANCE_CELL_FINDING,
    AbstractFieldDataset,
    AbstractMediumPropertyDataset,
    AuxFieldDataset,
    AuxFieldTimeDataset,
    Dataset,
    ElectromagneticFieldDataset,
    EMScalarFieldType,
    FieldDataset,
    FieldTimeDataset,
    FreqDataset,
    MediumDataset,
    ModeFreqDataset,
    ModeSolverDataset,
    PermittivityDataset,
    TimeDataset,
    TriangleMeshDataset,
)

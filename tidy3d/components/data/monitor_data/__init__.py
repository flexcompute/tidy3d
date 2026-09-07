"""Monitor data models and compatibility exports."""

from __future__ import annotations

from collections.abc import Sequence as _Sequence
from os import PathLike as _PathLike
from typing import Any as _Any

from tidy3d.components.data.data_array import (
    DataArray,
    DiffractionDataArray,
    DipoleEmissionDataArray,
    DipoleEmissionPositionDataArray,
    EMEFreqModeDataArray,
    FieldProjectionAngleDataArray,
    FieldProjectionCartesianDataArray,
    FieldProjectionKSpaceDataArray,
    FluxDataArray,
    FluxTimeDataArray,
    FreqDataArray,
    FreqModeDataArray,
    GroupIndexDataArray,
    MixedModeDataArray,
    ModeAmpsDataArray,
    ModeAmpsTimeDataArray,
    ModeDispersionDataArray,
    ModeIndexDataArray,
    ScalarFieldDataArray,
    TimeDataArray,
)
from tidy3d.components.geometry.base import Box as _Box
from tidy3d.components.grid.grid import Grid as _Grid
from tidy3d.components.types import (
    ArrayFloat2D as _ArrayFloat2D,
)
from tidy3d.components.types import (
    BoundOptional as _BoundOptional,
)
from tidy3d.components.types import (
    Coordinate as _Coordinate,
)

from ._constants import (
    AXIAL_RATIO_CAP,
    MIN_ANGULAR_SAMPLES_SPHERE,
    MODE_INTERP_EXTRAPOLATION_TOLERANCE,
    SHIFT_VALUE_ADJ_FLD_SRC,
)
from ._types import GRID_CORRECTION_TYPE, Coords1D, DataArrayCoordValue, DataArrayEntry, SourceT
from .base import AbstractFieldData, MonitorData
from .field import (
    AuxFieldTimeData,
    DipoleEmissionData,
    ElectromagneticFieldData,
    ElectromagneticSurfaceFieldData,
    FieldData,
    FieldStructureData,
    FieldTimeData,
    MediumData,
    PermittivityData,
    PointCloudFieldData,
    PointCloudPermittivityData,
    SurfaceFieldData,
    SurfaceFieldTimeData,
)
from .field import _em_algebra as _algebra_methods
from .field import _em_grid as _grid_methods
from .field import _em_io as _io_methods
from .field import _em_metrics as _metrics_methods
from .flux import FluxData, FluxTimeData
from .mode import AbstractOverlapData, FieldOverlapData, ModeData, ModeSolverData, ModeTimeData
from .projection import (
    AbstractFieldProjectionData,
    DiffractionData,
    DirectivityData,
    FieldProjectionAngleData,
    FieldProjectionCartesianData,
    FieldProjectionKSpaceData,
    ProjFieldType,
    ProjMonitorType,
)

# Register concrete classes after package imports so helper annotations remain
# resolvable without importing model modules from one another at runtime.
_algebra_methods.ArrayFloat2D = _ArrayFloat2D
_algebra_methods.DataArray = DataArray
_algebra_methods.ElectromagneticFieldData = ElectromagneticFieldData
_algebra_methods.FieldData = FieldData
_algebra_methods.ModeData = ModeData
_algebra_methods.ModeSolverData = ModeSolverData
_grid_methods.BoundOptional = _BoundOptional
_grid_methods.Coords1D = Coords1D
_grid_methods.ElectromagneticFieldData = ElectromagneticFieldData
_grid_methods.GRID_CORRECTION_TYPE = GRID_CORRECTION_TYPE
_grid_methods.Grid = _Grid
_io_methods.Coordinate = _Coordinate
_io_methods.ElectromagneticFieldData = ElectromagneticFieldData
_io_methods.FieldData = FieldData
_io_methods.PathLike = _PathLike
_io_methods.ScalarFieldDataArray = ScalarFieldDataArray
_metrics_methods.Any = _Any
_metrics_methods.Box = _Box
_metrics_methods.DataArray = DataArray
_metrics_methods.ElectromagneticFieldData = ElectromagneticFieldData
_metrics_methods.ScalarFieldDataArray = ScalarFieldDataArray
_metrics_methods.Sequence = _Sequence

__all__ = [
    "AXIAL_RATIO_CAP",
    "GRID_CORRECTION_TYPE",
    "MIN_ANGULAR_SAMPLES_SPHERE",
    "MODE_INTERP_EXTRAPOLATION_TOLERANCE",
    "SHIFT_VALUE_ADJ_FLD_SRC",
    "AbstractFieldData",
    "AbstractFieldProjectionData",
    "AbstractOverlapData",
    "AuxFieldTimeData",
    "Coords1D",
    "DataArray",
    "DataArrayCoordValue",
    "DataArrayEntry",
    "DiffractionData",
    "DiffractionDataArray",
    "DipoleEmissionData",
    "DipoleEmissionDataArray",
    "DipoleEmissionPositionDataArray",
    "DirectivityData",
    "EMEFreqModeDataArray",
    "ElectromagneticFieldData",
    "ElectromagneticSurfaceFieldData",
    "FieldData",
    "FieldOverlapData",
    "FieldProjectionAngleData",
    "FieldProjectionAngleDataArray",
    "FieldProjectionCartesianData",
    "FieldProjectionCartesianDataArray",
    "FieldProjectionKSpaceData",
    "FieldProjectionKSpaceDataArray",
    "FieldStructureData",
    "FieldTimeData",
    "FluxData",
    "FluxDataArray",
    "FluxTimeData",
    "FluxTimeDataArray",
    "FreqDataArray",
    "FreqModeDataArray",
    "GroupIndexDataArray",
    "MediumData",
    "MixedModeDataArray",
    "ModeAmpsDataArray",
    "ModeAmpsTimeDataArray",
    "ModeData",
    "ModeDispersionDataArray",
    "ModeIndexDataArray",
    "ModeSolverData",
    "ModeTimeData",
    "MonitorData",
    "PermittivityData",
    "PointCloudFieldData",
    "PointCloudPermittivityData",
    "ProjFieldType",
    "ProjMonitorType",
    "ScalarFieldDataArray",
    "SourceT",
    "SurfaceFieldData",
    "SurfaceFieldTimeData",
    "TimeDataArray",
]

"""Type definitions for monitor data."""

from __future__ import annotations

from tidy3d.components.data.monitor_data import (
    AuxFieldTimeData,
    DiffractionData,
    DirectivityData,
    FieldData,
    FieldOverlapData,
    FieldProjectionAngleData,
    FieldProjectionCartesianData,
    FieldProjectionKSpaceData,
    FieldTimeData,
    FluxData,
    FluxTimeData,
    MediumData,
    ModeData,
    ModeSolverData,
    PermittivityData,
    SurfaceFieldData,
    SurfaceFieldTimeData,
)
from tidy3d.components.microwave.data.monitor_data import MicrowaveModeData, MicrowaveModeSolverData

# Type aliases
ModeDataType = ModeData | MicrowaveModeData
ModeSolverDataType = ModeSolverData | MicrowaveModeSolverData
MonitorDataTypes = (
    FieldData,
    FieldTimeData,
    PermittivityData,
    MediumData,
    ModeSolverData,
    ModeData,
    FluxData,
    FluxTimeData,
    AuxFieldTimeData,
    FieldProjectionKSpaceData,
    FieldProjectionCartesianData,
    FieldProjectionAngleData,
    DiffractionData,
    DirectivityData,
    FieldOverlapData,
    MicrowaveModeData,
    MicrowaveModeSolverData,
    SurfaceFieldData,
    SurfaceFieldTimeData,
)
MonitorDataType = (
    FieldData
    | FieldTimeData
    | PermittivityData
    | MediumData
    | ModeSolverData
    | ModeData
    | FluxData
    | FluxTimeData
    | AuxFieldTimeData
    | FieldProjectionKSpaceData
    | FieldProjectionCartesianData
    | FieldProjectionAngleData
    | DiffractionData
    | DirectivityData
    | FieldOverlapData
    | MicrowaveModeData
    | MicrowaveModeSolverData
    | SurfaceFieldData
    | SurfaceFieldTimeData
)
